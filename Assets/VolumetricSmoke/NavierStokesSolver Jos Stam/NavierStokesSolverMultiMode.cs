using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NavierStokesSolverMultiMode : MonoBehaviour    // referance paper : https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
{
    public static HashSet<NavierStokesSolverMultiMode> all_instances = new HashSet<NavierStokesSolverMultiMode>();

    public void OnEnable()
    {
        all_instances.Add(this);
    }

    public void OnDisable()
    {
        all_instances.Remove(this);
    }

    // simulation parameters
    private int size;
    private int iter;

    public Vector3 gravity = new Vector3(0, -9.8f, 0);

    private float dt;
    private float diff;
    private float visc;

    // CPU mode buffers
    private float[] s;
    private float[] density;

    private float[] Vx;
    private float[] Vy;

    private float[] Vx0;
    private float[] Vy0;

    private bool[] boundries;

    // GPU mode buffers
    /*[HideInInspector]*/
    public RenderTexture s_rt;
    /*[HideInInspector]*/
    public RenderTexture density_rt;

    /*[HideInInspector]*/
    public RenderTexture Vx_rt;
    /*[HideInInspector]*/
    public RenderTexture Vy_rt;
    /*[HideInInspector]*/
    public RenderTexture Vz_rt;

    /*[HideInInspector]*/
    public RenderTexture Vx0_rt;
    /*[HideInInspector]*/
    public RenderTexture Vy0_rt;
    /*[HideInInspector]*/
    public RenderTexture Vz0_rt;

    /*[HideInInspector]*/
    public RenderTexture boundries_rt;

    public ComputeShader solver;
    public ComputeShader solver3D;
    private Vector3 solver3D_kernel_size;

    // flags
    private bool _last_iteration_was_cpu = false;
    private bool _cpu_buffers_initialized = false;
    private bool _gpu_buffers_initialized = false;
    private bool _gpu_buffers_sized_for_3D = false;

    public float target_frametime_ms;
    private float carryover_time_ms;

    public enum ComputationMode
    {
        GPU3D,
        GPU2D,
        CPU2D,
        COMPUTE_DEBUG2D,
    }

    public ComputationMode mode;

    public enum GPULinearSolver
    {
        JACOBI,
        GAUSS_SEISEL
    }

    public GPULinearSolver linear_solver_method;

    public NavierStokesSolverMultiMode(int size, float dt, int iterations, float diffusion, float viscosity)
    {
        init_parameters(size, iterations, dt, diffusion, viscosity);
        init_cpu();
        init_gpu();
    }

    private void free_vram()
    {
        if (s_rt != null) s_rt.Release();
        if (density_rt != null) density_rt.Release();
        if (Vx_rt != null) Vx_rt.Release();
        if (Vy_rt != null) Vy_rt.Release();
        if (Vz_rt != null) Vz_rt.Release();
        if (Vx0_rt != null) Vx0_rt.Release();
        if (Vy0_rt != null) Vy0_rt.Release();
        if (Vz0_rt != null) Vz0_rt.Release();
        if (temp_linear_solve_jacobi != null) temp_linear_solve_jacobi.Release();
        if (temp_advect_vx != null) temp_advect_vx.Release();
        if (temp_advect_vy != null) temp_advect_vy.Release();
        if (temp_advect_vz != null) temp_advect_vz.Release();
        if (slice_of_3D != null) slice_of_3D.Release();
    }

    public void OnDestroy()
    {
        free_vram();
    }

    /// <summary>
    /// initialize simulation parameters
    /// </summary>
    public void init_parameters(int size, int iterations, float dt, float diffusion, float viscosity)
    {
        solver3D_kernel_size = new Vector3(2, 2, 2);

        if (size != this.size)
        {
            _last_iteration_was_cpu = false;
            _cpu_buffers_initialized = false;
            _gpu_buffers_initialized = false;
            _first_draw_in_cpu_mode = true;
            _first_draw_in_gpu_mode = true;
            _gpu_buffers_sized_for_3D = false;
        }

        iter = iterations;
        this.size = size;
        this.dt = dt;
        this.diff = diffusion;
        this.visc = viscosity;
    }

    /// <summary>
    /// initialize CPU buffers
    /// </summary>
    public void init_cpu()
    {
        if (_cpu_buffers_initialized) return;
        _cpu_buffers_initialized = true;

        this.s = new float[size * size];
        this.density = new float[size * size];

        this.Vx = new float[size * size];
        this.Vy = new float[size * size];

        this.Vx0 = new float[size * size];
        this.Vy0 = new float[size * size];

        this.boundries = new bool[size * size];
    }

    /// <summary>
    /// initialize GPU buffers
    /// </summary>
    public void init_gpu()
    {
        bool dimention_mismatch = (mode == ComputationMode.GPU3D && !_gpu_buffers_sized_for_3D) || (mode == ComputationMode.GPU2D && _gpu_buffers_sized_for_3D);
        bool should_initialize = !_gpu_buffers_initialized || dimention_mismatch;

        if (!should_initialize) return;
        _gpu_buffers_initialized = true;

        if (mode == ComputationMode.GPU3D)
        {
            _gpu_buffers_sized_for_3D = true;

            Debug.Log("3d gpu init");

            this.s_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.density_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vx_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vy_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vz_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vx0_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vy0_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vz0_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.boundries_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);

            this.s_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.density_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vx_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vy_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vz_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vx0_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vy0_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vz0_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.boundries_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;

            this.s_rt.volumeDepth = size;
            this.density_rt.volumeDepth = size;
            this.Vx_rt.volumeDepth = size;
            this.Vy_rt.volumeDepth = size;
            this.Vz_rt.volumeDepth = size;
            this.Vx0_rt.volumeDepth = size;
            this.Vy0_rt.volumeDepth = size;
            this.Vz0_rt.volumeDepth = size;
            this.boundries_rt.volumeDepth = size;

            this.s_rt.wrapMode = TextureWrapMode.Clamp;
            this.density_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vx_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vy_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vz_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vx0_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vy0_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vz0_rt.wrapMode = TextureWrapMode.Clamp;
            this.boundries_rt.wrapMode = TextureWrapMode.Clamp;

            this.s_rt.enableRandomWrite = true;
            this.density_rt.enableRandomWrite = true;
            this.Vx_rt.enableRandomWrite = true;
            this.Vy_rt.enableRandomWrite = true;
            this.Vz_rt.enableRandomWrite = true;
            this.Vx0_rt.enableRandomWrite = true;
            this.Vy0_rt.enableRandomWrite = true;
            this.Vz0_rt.enableRandomWrite = true;
            this.boundries_rt.enableRandomWrite = true;

            this.s_rt.filterMode = FilterMode.Bilinear;
            this.density_rt.filterMode = FilterMode.Bilinear;
            this.Vx_rt.filterMode = FilterMode.Bilinear;
            this.Vy_rt.filterMode = FilterMode.Bilinear;
            this.Vz_rt.filterMode = FilterMode.Bilinear;
            this.Vx0_rt.filterMode = FilterMode.Bilinear;
            this.Vy0_rt.filterMode = FilterMode.Bilinear;
            this.Vz0_rt.filterMode = FilterMode.Bilinear;
            this.boundries_rt.filterMode = FilterMode.Bilinear;
        }
        else
        {
            _gpu_buffers_sized_for_3D = false;

            Debug.Log("2d gpu init");

            this.s_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.density_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vx_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vy_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vz_rt = null;
            this.Vx0_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vy0_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            this.Vz0_rt = null;
            this.boundries_rt = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);

            this.s_rt.wrapMode = TextureWrapMode.Clamp;
            this.density_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vx_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vy_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vx0_rt.wrapMode = TextureWrapMode.Clamp;
            this.Vy0_rt.wrapMode = TextureWrapMode.Clamp;
            this.boundries_rt.wrapMode = TextureWrapMode.Clamp;

            this.s_rt.enableRandomWrite = true;
            this.density_rt.enableRandomWrite = true;
            this.Vx_rt.enableRandomWrite = true;
            this.Vy_rt.enableRandomWrite = true;
            this.Vx0_rt.enableRandomWrite = true;
            this.Vy0_rt.enableRandomWrite = true;
            this.boundries_rt.enableRandomWrite = true;

            this.s_rt.filterMode = FilterMode.Bilinear;
            this.density_rt.filterMode = FilterMode.Bilinear;
            this.Vx_rt.filterMode = FilterMode.Bilinear;
            this.Vy_rt.filterMode = FilterMode.Bilinear;
            this.Vx0_rt.filterMode = FilterMode.Bilinear;
            this.Vy0_rt.filterMode = FilterMode.Bilinear;
            this.boundries_rt.filterMode = FilterMode.Bilinear;
        }

    }

    #region UTILS

    /// <summary>
    /// synchronization of buffers between GPU and CPU, computationally expensive, only needed when simulation mode is changed from GPU to CPU
    /// </summary>
    private void transfer_data_from_cpu_to_gpu()
    {
        init_cpu();
        init_gpu();
        array_to_texture(Vx, Vx_rt);
        array_to_texture(Vx0, Vx0_rt);
        array_to_texture(Vy, Vy_rt);
        array_to_texture(Vy0, Vy0_rt);
        array_to_texture(s, s_rt);
        array_to_texture(density, density_rt);
    }

    /// <summary>
    /// synchronization of buffers between GPU and CPU, computationally expensive, only needed when simulation mode is changed from CPU to GPU
    /// </summary>
    private void transfer_data_from_gpu_to_cpu()
    {
        init_cpu();
        init_gpu();
        texture_to_array(Vx_rt, Vx);
        texture_to_array(Vx0_rt, Vx0);
        texture_to_array(Vy_rt, Vy);
        texture_to_array(Vy0_rt, Vy0);
        texture_to_array(s_rt, s);
        texture_to_array(density_rt, density);
    }


    /// <summary>
    /// add the given amount to a single index of a GPU buffer
    /// </summary>
    private void _write_to_texture(RenderTexture target, int x, int y, float amount)
    {
        int kernel_index = solver.FindKernel("write_to_texture");
        solver.SetVector("write_position", new Vector2(x, y));
        solver.SetFloat("value", amount);
        solver.SetTexture(kernel_index, "write_target", target);
        solver.Dispatch(kernel_index, 1, 1, 1);
    }

    /// <summary>
    /// add the given amount to a single index of a GPU3D buffer
    /// </summary>
    private void _write_to_texture_3d(RenderTexture target, int x, int y, int z, float amount)
    {
        float write_to_texture_begin_time = Time.realtimeSinceStartup * 1000;

        int kernel_index = solver3D.FindKernel("write_to_texture_3d");
        solver3D.SetVector("write_position", new Vector3(x, y, z));
        solver3D.SetFloat("value", amount);
        solver3D.SetTexture(kernel_index, "write_target", target);
        solver3D.Dispatch(kernel_index, 1, 1, 1);

        write_to_texture_total_cost_3d += Time.realtimeSinceStartup * 1000 - write_to_texture_begin_time;
        write_to_texture_call_count_3d++;
    }

    /// <summary>
    /// convert CPU buffer to GPU buffer
    /// </summary>
    private RenderTexture array_to_texture(float[] array)
    {
        ComputeBuffer buffer = new ComputeBuffer(size * size, sizeof(float), ComputeBufferType.Structured);
        buffer.SetData(array);

        RenderTexture result = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
        result.enableRandomWrite = true;
        result.filterMode = FilterMode.Bilinear;

        int kernel_index = solver.FindKernel("copy_to_texture");
        solver.SetInt("size", size);
        solver.SetBuffer(kernel_index, "copy_source_buffer", buffer);
        solver.SetTexture(kernel_index, "copy_target_texture", result);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        buffer.Dispose();

        return result;
    }

    /// <summary>
    /// convert CPU buffer to GPU buffer
    /// </summary>
    private void array_to_texture(float[] array, RenderTexture output_texture)
    {
        ComputeBuffer buffer = new ComputeBuffer(size * size, sizeof(float), ComputeBufferType.Structured);
        buffer.SetData(array);

        int kernel_index = solver.FindKernel("copy_to_texture");
        solver.SetInt("size", size);
        solver.SetBuffer(kernel_index, "copy_source_buffer", buffer);
        solver.SetTexture(kernel_index, "copy_target_texture", output_texture);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        buffer.Dispose();
    }

    /// <summary>
    /// convert GPU buffer to CPU buffer
    /// </summary>
    private float[] texture_to_array(RenderTexture texture)
    {
        ComputeBuffer buffer = new ComputeBuffer(size * size, sizeof(float), ComputeBufferType.Structured);

        int kernel_index = solver.FindKernel("copy_to_buffer");
        solver.SetInt("size", size);
        solver.SetTexture(kernel_index, "copy_source_texture", texture);
        solver.SetBuffer(kernel_index, "copy_target_buffer", buffer);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        float[] data = new float[size * size];
        buffer.GetData(data);

        buffer.Dispose();

        return data;
    }

    /// <summary>
    /// convert GPU buffer to CPU buffer
    /// </summary>
    private void texture_to_array(RenderTexture texture, float[] output_array)
    {
        ComputeBuffer buffer = new ComputeBuffer(size * size, sizeof(float), ComputeBufferType.Structured);

        int kernel_index = solver.FindKernel("copy_to_buffer");
        solver.SetInt("size", size);
        solver.SetTexture(kernel_index, "copy_source_texture", texture);
        solver.SetBuffer(kernel_index, "copy_target_buffer", buffer);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        buffer.GetData(output_array);

        buffer.Dispose();
    }

    #endregion

    #region DEBUG

    /// <summary>
    /// single iteration of simulation in COMPUTE mode, this mode still uses CPU buffers but uses GPU for computation, poor performance and only to be used for debugging
    /// </summary>
    private void step_compute()
    {
        _last_iteration_was_cpu = true;

        init_cpu();
        init_gpu();

        add_external_force_compute(gravity.x * 0.0001f, Vx);
        add_external_force_compute(gravity.y * 0.0001f, Vy);

        diffuse_compute(1, Vx0, Vx);
        diffuse_compute(2, Vy0, Vy);

        project_compute(Vx0, Vy0, Vx, Vy);

        advect_compute(1, Vx, Vx0, Vx0, Vy0);
        advect_compute(2, Vy, Vy0, Vx0, Vy0);

        project_compute(Vx, Vy, Vx0, Vy0);

        diffuse_compute(0, s, density);
        advect_compute(0, density, s, Vx, Vy);
    }

    /// <summary>
    /// add density to CPU buffer, use GPU algorithm to do so, only for debugging
    /// </summary>
    private void add_density_compute(int x, int y, float amount)
    {
        init_cpu();
        init_gpu();

        RenderTexture density_texture = array_to_texture(density);
        _write_to_texture(density_texture, x, y, amount);
        texture_to_array(density_texture, density);
    }

    /// <summary>
    /// add velocity to CPU buffer, use GPU algorithm to do so, only for debugging
    /// </summary>
    private void add_velocity_compute(int x, int y, float amount_x, float amount_y)
    {
        init_cpu();
        init_gpu();

        RenderTexture vx_t = array_to_texture(Vx);
        RenderTexture vy_t = array_to_texture(Vy);

        _write_to_texture(vx_t, x, y, amount_x);
        _write_to_texture(vy_t, x, y, amount_y);

        texture_to_array(vx_t, Vx);
        texture_to_array(vy_t, Vy);
    }

    private void _set_boundry_compute_internal(int x, int y, float amount)
    {
        float[] boundries_float = new float[size * size];
        for (int i = 0; i < size * size; i++)
            boundries_float[i] = boundries[i] ? 1.0f : 0.0f;

        RenderTexture b_t = array_to_texture(boundries_float);
        _write_to_texture(b_t, x, y, amount);
        texture_to_array(b_t, boundries_float);

        for (int i = 0; i < size * size; i++)
            boundries[i] = boundries_float[i] > 0.5f;
    }

    /// <summary>
    /// set given rectangular area as boundry, CPU mode, use GPU for computation, only for debugging
    /// </summary>
    private void set_boundry_compute(int x0, int y0, int size_x, int size_y)
    {
        init_cpu();
        init_gpu();

        int new_boundry = 0;
        for (int x = x0; x < x0 + size_x; x++)
        {
            for (int y = y0; y < y0 + size_y; y++)
            {
                _set_boundry_compute_internal(x, y, 1.0f);
                new_boundry++;
            }
        }
        //Debug.Log(new_boundry + " new boundries added");
    }

    /// <summary>
    /// simulation behaviour at boundries, CPU mode, use GPU algorithm in single thread for computation, poor performance and only for debuging
    /// </summary>
    private void set_bnd_compute_single(int b, float[] x)
    {
        init_cpu();
        init_gpu();

        float[] boundries_float = new float[size * size];
        for (int i = 0; i < size * size; i++)
            boundries_float[i] = boundries[i] ? 1.0f : 0.0f;
        RenderTexture boundries_t = array_to_texture(boundries_float);

        RenderTexture x_t = array_to_texture(x);

        int kernel_index = solver.FindKernel("set_bnd_single");
        solver.SetInt("size", size);
        solver.SetInt("b", b);
        solver.SetTexture(kernel_index, "x", x_t);
        solver.SetTexture(kernel_index, "boundries", boundries_t);
        solver.Dispatch(kernel_index, 1, 1, 1);

        float[] x_result = texture_to_array(x_t);
        for (int i = 0; i < size * size; i++)
            x[i] = x_result[i];
    }

    /// <summary>
    /// simulation behaviour at boundries, CPU mode, use GPU algorithm for computation, poor performance and only for debuging
    /// </summary>
    private void set_bnd_compute(int b, float[] x)
    {
        init_cpu();
        init_gpu();

        RenderTexture x_t = array_to_texture(x);

        set_bnd_gpu2d(b, x_t);

        float[] x_result = texture_to_array(x_t);
        for (int i = 0; i < size * size; i++)
            x[i] = x_result[i];
    }

    /// <summary>
    /// simulate diffusion, CPU mode, use GPU algorithm in single thread for computation, poor performance and only for debuging
    /// </summary>
    private void diffuse_compute_single(int b, float[] x, float[] x0)
    {
        init_cpu();
        init_gpu();

        float a = dt * diff * (size - 2) * (size - 2);
        lin_solve_compute_single(b, x, x0, a, 1 + 4 * a);
    }

    /// <summary>
    /// simulate diffusion, CPU mode, use GPU algorithm for computation, poor performance and only for debuging
    /// </summary>
    private void diffuse_compute(int b, float[] x, float[] x0)
    {
        init_cpu();
        init_gpu();

        float a = dt * diff * (size - 2) * (size - 2);
        lin_solve_compute(b, x, x0, a, 1 + 4 * a);
    }

    /// <summary>
    /// solve linear system, CPU mode, use GPU algorithm in single thread for computation, poor performance and only for debuging
    /// </summary>
    private void lin_solve_compute_single(int b, float[] x, float[] x0, float a, float c)
    {
        init_cpu();
        init_gpu();

        RenderTexture x_t = null;
        RenderTexture x0_t = null;

        for (int i = 0; i < iter; i++)
        {
            x_t = array_to_texture(x);
            x0_t = array_to_texture(x0);

            int kernel_index = solver.FindKernel("lin_solve_single");
            solver.SetInt("size", size);
            solver.SetInt("b", b);
            solver.SetFloat("a", a);
            solver.SetFloat("c", c);
            solver.SetTexture(kernel_index, "x", x_t);
            solver.SetTexture(kernel_index, "x0", x0_t);
            solver.SetTexture(kernel_index, "x_old", x_t);
            solver.Dispatch(kernel_index, 1, 1, 1);

            float[] x_result_iteration = texture_to_array(x_t);
            float[] x0_result_iteration = texture_to_array(x0_t);
            for (int j = 0; j < size * size; j++)
            {
                x[j] = x_result_iteration[j];
                x0[j] = x0_result_iteration[j];
            }
            set_bnd_compute(b, x);
        }

        float[] x_result = texture_to_array(x_t);
        float[] x0_result = texture_to_array(x0_t);
        for (int i = 0; i < size * size; i++)
        {
            x[i] = x_result[i];
            x0[i] = x0_result[i];
        }

    }

    /// <summary>
    /// solve linear system, CPU mode, use GPU algorithm for computation, poor performance and only for debuging
    /// </summary>
    private void lin_solve_compute(int b, float[] x, float[] x0, float a, float c)
    {
        init_cpu();
        init_gpu();

        RenderTexture x_t = array_to_texture(x);
        RenderTexture x0_t = array_to_texture(x0);

        RenderTexture temp = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
        temp.enableRandomWrite = true;
        temp.filterMode = FilterMode.Bilinear;

        for (int i = 0; i < iter; i++)
        {
            int kernel_index = solver.FindKernel("copy_textures");
            solver.SetTexture(kernel_index, "copy_source_texture", x_t);
            solver.SetTexture(kernel_index, "copy_target_texture", temp);
            solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

            kernel_index = solver.FindKernel("lin_solve");
            solver.SetInt("size", size);
            solver.SetInt("b", b);
            solver.SetFloat("a", a);
            solver.SetFloat("c", c);
            solver.SetTexture(kernel_index, "x", x_t);
            solver.SetTexture(kernel_index, "x0", x0_t);
            solver.SetTexture(kernel_index, "x_old", temp);
            solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

            float[] x_result_iteration = texture_to_array(x_t);
            float[] x0_result_iteration = texture_to_array(x0_t);
            for (int j = 0; j < size * size; j++)
            {
                x[j] = x_result_iteration[j];
                x0[j] = x0_result_iteration[j];
            }
            set_bnd_compute(b, x);
        }

        float[] x_result = texture_to_array(x_t);
        float[] x0_result = texture_to_array(x0_t);
        for (int i = 0; i < size * size; i++)
        {
            x[i] = x_result[i];
            x0[i] = x0_result[i];
        }
    }

    /// <summary>
    /// add force to the entirety of space, CPU mode, use GPU algorithm for computation, poor performance and only for debuging
    /// </summary>
    private void add_external_force_compute(float force, float[] d)
    {
        init_cpu();
        init_gpu();

        RenderTexture d_t = array_to_texture(d);

        add_external_force_gpu2d(force, d_t);

        float[] d_reuslt = texture_to_array(d_t);
        for (int i = 0; i < size * size; i++)
            d[i] = d_reuslt[i];
    }

    /// <summary>
    /// regulate velocities to achieve zero divergence everywhere, CPU mode, use GPU algorithm in single thread for computation, poor performance and only for debuging
    /// </summary>
    private void project_compute_single(float[] velocX, float[] velocY, float[] p, float[] div)
    {
        init_cpu();
        init_gpu();

        RenderTexture vx_t = array_to_texture(velocX);
        RenderTexture vy_t = array_to_texture(velocY);
        RenderTexture p_t = array_to_texture(p);
        RenderTexture div_t = array_to_texture(div);

        int kernel_index = solver.FindKernel("project_1_single");
        solver.SetInt("size", size);
        solver.SetTexture(kernel_index, "p", p_t);
        solver.SetTexture(kernel_index, "div", div_t);
        solver.SetTexture(kernel_index, "velocity_x_1", vx_t);
        solver.SetTexture(kernel_index, "velocity_y_1", vy_t);
        solver.Dispatch(kernel_index, 1, 1, 1);

        float[] p_result = texture_to_array(p_t);
        float[] div_result = texture_to_array(div_t);
        float[] vx_result = texture_to_array(vx_t);
        float[] vy_result = texture_to_array(vy_t);

        for (int i = 0; i < size * size; i++)
        {
            p[i] = p_result[i];
            div[i] = div_result[i];
            velocX[i] = vx_result[i];
            velocY[i] = vy_result[i];
        }

        set_bnd_compute_single(0, div);
        set_bnd_compute_single(0, p);
        lin_solve_compute_single(0, p, div, 1, 4);

        vx_t = array_to_texture(velocX);
        vy_t = array_to_texture(velocY);
        p_t = array_to_texture(p);
        div_t = array_to_texture(div);

        kernel_index = solver.FindKernel("project_2_single");
        solver.SetInt("size", size);
        solver.SetTexture(kernel_index, "p", p_t);
        solver.SetTexture(kernel_index, "div", div_t);
        solver.SetTexture(kernel_index, "velocity_x_1", vx_t);
        solver.SetTexture(kernel_index, "velocity_y_1", vy_t);
        solver.Dispatch(kernel_index, 1, 1, 1);

        p_result = texture_to_array(p_t);
        div_result = texture_to_array(div_t);
        vx_result = texture_to_array(vx_t);
        vy_result = texture_to_array(vy_t);

        for (int i = 0; i < size * size; i++)
        {
            p[i] = p_result[i];
            div[i] = div_result[i];
            velocX[i] = vx_result[i];
            velocY[i] = vy_result[i];
        }

        set_bnd_compute_single(1, velocX);
        set_bnd_compute_single(2, velocY);
    }

    /// <summary>
    /// regulate velocities to achieve zero divergence everywhere, CPU mode, use GPU algorithm for computation, poor performance and only for debuging
    /// </summary>
    private void project_compute(float[] velocX, float[] velocY, float[] p, float[] div)
    {
        init_cpu();
        init_gpu();

        RenderTexture vx_t = array_to_texture(velocX);
        RenderTexture vy_t = array_to_texture(velocY);
        RenderTexture p_t = array_to_texture(p);
        RenderTexture div_t = array_to_texture(div);

        int kernel_index = solver.FindKernel("project_1");
        solver.SetInt("size", size);
        solver.SetTexture(kernel_index, "p", p_t);
        solver.SetTexture(kernel_index, "div", div_t);
        solver.SetTexture(kernel_index, "velocity_x_1", vx_t);
        solver.SetTexture(kernel_index, "velocity_y_1", vy_t);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        float[] p_result = texture_to_array(p_t);
        float[] div_result = texture_to_array(div_t);
        float[] vx_result = texture_to_array(vx_t);
        float[] vy_result = texture_to_array(vy_t);

        for (int i = 0; i < size * size; i++)
        {
            p[i] = p_result[i];
            div[i] = div_result[i];
            velocX[i] = vx_result[i];
            velocY[i] = vy_result[i];
        }

        set_bnd_compute(0, div);
        set_bnd_compute(0, p);
        lin_solve_compute(0, p, div, 1, 4);

        vx_t = array_to_texture(velocX);
        vy_t = array_to_texture(velocY);
        p_t = array_to_texture(p);
        div_t = array_to_texture(div);

        kernel_index = solver.FindKernel("project_2");
        solver.SetInt("size", size);
        solver.SetTexture(kernel_index, "p", p_t);
        solver.SetTexture(kernel_index, "div", div_t);
        solver.SetTexture(kernel_index, "velocity_x_1", vx_t);
        solver.SetTexture(kernel_index, "velocity_y_1", vy_t);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        p_result = texture_to_array(p_t);
        div_result = texture_to_array(div_t);
        vx_result = texture_to_array(vx_t);
        vy_result = texture_to_array(vy_t);

        for (int i = 0; i < size * size; i++)
        {
            p[i] = p_result[i];
            div[i] = div_result[i];
            velocX[i] = vx_result[i];
            velocY[i] = vy_result[i];
        }

        set_bnd_compute(1, velocX);
        set_bnd_compute(2, velocY);
    }

    /// <summary>
    /// flow given field in the direction of velocity fields, CPU mode, use GPU algorithm in single thread for computation, poor performance and only for debuging
    /// </summary>
    private void advect_compute_single(int b, float[] d, float[] d0, float[] velocX, float[] velocY)
    {
        init_cpu();
        init_gpu();

        RenderTexture d_t = array_to_texture(d);
        RenderTexture d0_t = array_to_texture(d0);
        RenderTexture vx_t = array_to_texture(velocX);
        RenderTexture vy_t = array_to_texture(velocY);

        int kernel_index = solver.FindKernel("advect_single");
        solver.SetInt("size", size);
        solver.SetInt("b", b);
        solver.SetFloat("dt", dt);
        solver.SetTexture(kernel_index, "d", d_t);
        solver.SetTexture(kernel_index, "d0", d0_t);
        solver.SetTexture(kernel_index, "velocity_x_1", vx_t);
        solver.SetTexture(kernel_index, "velocity_y_1", vy_t);
        solver.Dispatch(kernel_index, 1, 1, 1);

        float[] d_result = texture_to_array(d_t);
        float[] d0_result = texture_to_array(d0_t);
        float[] vx_result = texture_to_array(vx_t);
        float[] vy_result = texture_to_array(vy_t);

        for (int i = 0; i < size * size; i++)
        {
            d[i] = d_result[i];
            d0[i] = d0_result[i];
            velocX[i] = vx_result[i];
            velocY[i] = vy_result[i];
        }

        set_bnd_compute(b, d);
    }

    /// <summary>
    /// flow given field in the direction of velocity fields, CPU mode, use GPU algorithm for computation, poor performance and only for debuging
    /// </summary>
    private void advect_compute(int b, float[] d, float[] d0, float[] velocX, float[] velocY)
    {
        init_cpu();
        init_gpu();

        RenderTexture d_t = array_to_texture(d);
        RenderTexture d0_t = array_to_texture(d0);
        RenderTexture vx_t = array_to_texture(velocX);
        RenderTexture vy_t = array_to_texture(velocY);

        int kernel_index = solver.FindKernel("advect");
        solver.SetInt("size", size);
        solver.SetInt("b", b);
        solver.SetFloat("dt", dt);
        solver.SetTexture(kernel_index, "d", d_t);
        solver.SetTexture(kernel_index, "d0", d0_t);
        solver.SetTexture(kernel_index, "velocity_x_1", vx_t);
        solver.SetTexture(kernel_index, "velocity_y_1", vy_t);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        float[] d_result = texture_to_array(d_t);
        float[] d0_result = texture_to_array(d0_t);
        float[] vx_result = texture_to_array(vx_t);
        float[] vy_result = texture_to_array(vy_t);

        for (int i = 0; i < size * size; i++)
        {
            d[i] = d_result[i];
            d0[i] = d0_result[i];
            velocX[i] = vx_result[i];
            velocY[i] = vy_result[i];
        }

        set_bnd_compute(b, d);
    }

    #endregion

    #region CPU

    /// <summary>
    /// in CPU buffers are stored as 1D arrays, this functions converts 2D indexing format to 1D indexing
    /// </summary>
    private int IX(int x, int y)
    {
        if (x > size - 1) x = size - 1;
        if (y > size - 1) y = size - 1;
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        return x + (y * size);
    }

    /// <summary>
    /// single iteration of simulation in CPU mode
    /// </summary>
    private void step_cpu()
    {
        _last_iteration_was_cpu = true;

        init_cpu();

        add_external_force_cpu(gravity.x * 0.0001f, Vx);
        add_external_force_cpu(gravity.y * 0.0001f, Vy);

        diffuse_cpu(1, Vx0, Vx);
        diffuse_cpu(2, Vy0, Vy);

        project_cpu(Vx0, Vy0, Vx, Vy);

        advect_cpu(1, Vx, Vx0, Vx0, Vy0);
        advect_cpu(2, Vy, Vy0, Vx0, Vy0);

        project_cpu(Vx, Vy, Vx0, Vy0);

        diffuse_cpu(0, s, density);
        advect_cpu(0, density, s, Vx, Vy);
    }

    /// <summary>
    /// add density to CPU buffer
    /// </summary>
    private void add_density_cpu(int x, int y, float amount)
    {
        init_cpu();

        int index = IX(x, y);
        if (boundries[index]) return;
        this.density[index] += amount;
    }

    /// <summary>
    /// add velocity to CPU buffer
    /// </summary>
    private void add_velocity_cpu(int x, int y, float amountX, float amountY)
    {
        init_cpu();

        int index = IX(x, y);
        if (boundries[index]) return;
        this.Vx[index] += amountX;
        this.Vy[index] += amountY;
    }

    /// <summary>
    /// set given rectangular area as boundry, CPU mode
    /// </summary>
    private void set_boundry_cpu(int x0, int y0, int size_x, int size_y)
    {
        init_cpu();

        int new_boundry = 0;
        for (int x = x0; x < x0 + size_x; x++)
        {
            for (int y = y0; y < y0 + size_y; y++)
            {
                boundries[IX(x, y)] = true;
                new_boundry++;
            }
        }
        //Debug.Log(new_boundry + " new boundries added");
    }

    /// <summary>
    /// simulation behaviour at boundries, CPU mode
    /// </summary>
    private void set_bnd_cpu(int b, float[] x)
    {
        init_cpu();

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (boundries[IX(i, j)])
                {
                    int non_boundry_neigbour_count = 0;
                    float sum = 0;
                    if (!boundries[IX(i + 1, j)])
                    {
                        non_boundry_neigbour_count++;
                        sum += b == 1 ? -x[IX(i + 1, j)] : x[IX(i + 1, j)];
                    }
                    if (!boundries[IX(i - 1, j)])
                    {
                        non_boundry_neigbour_count++;
                        sum += b == 1 ? -x[IX(i - 1, j)] : x[IX(i - 1, j)];
                    }

                    if (!boundries[IX(i, j + 1)])
                    {
                        non_boundry_neigbour_count++;
                        sum += b == 2 ? -x[IX(i, j + 1)] : x[IX(i, j + 1)];
                    }
                    if (!boundries[IX(i, j - 1)])
                    {
                        non_boundry_neigbour_count++;
                        sum += b == 2 ? -x[IX(i, j - 1)] : x[IX(i, j - 1)];
                    }

                    if (non_boundry_neigbour_count == 0) continue;

                    sum = sum / non_boundry_neigbour_count;
                    x[IX(i, j)] = sum;
                }
            }
        }

        //for (int i = 1; i < size - 1; i++)
        //{
        //    x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        //    x[IX(i, size - 1)] = b == 2 ? -x[IX(i, size - 2)] : x[IX(i, size - 2)];
        //}
        //for (int j = 1; j < size - 1; j++)
        //{
        //    x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
        //    x[IX(size - 1, j)] = b == 1 ? -x[IX(size - 2, j)] : x[IX(size - 2, j)];
        //}

        //x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
        //x[IX(0, size - 1)] = 0.5f * (x[IX(1, size - 1)] + x[IX(0, size - 2)]);
        //x[IX(size - 1, 0)] = 0.5f * (x[IX(size - 2, 0)] + x[IX(size - 1, 1)]);
        //x[IX(size - 1, size - 1)] = 0.5f * (x[IX(size - 2, size - 1)] + x[IX(size - 1, size - 2)]);
    }

    /// <summary>
    /// simulate diffusion, CPU mode
    /// </summary>
    private void diffuse_cpu(int b, float[] x, float[] x0)
    {
        init_cpu();

        float a = dt * diff * (size - 2) * (size - 2);
        lin_solve_cpu(b, x, x0, a, 1 + 4 * a);
    }

    /// <summary>
    /// solve linear system, CPU mode
    /// </summary>
    private void lin_solve_cpu(int b, float[] x, float[] x0, float a, float c)
    {
        init_cpu();

        float cRecip = 1.0f / c;
        //float[] old_x = new float[size * size];
        float[] old_x = x;

        for (int k = 0; k < iter; k++)
        {
            //for (int i = 0; i < size * size; i++)
            //{
            //    old_x[i] = x[i];
            //}

            for (int j = 1; j < size - 1; j++)
            {
                for (int i = 1; i < size - 1; i++)
                {
                    x[IX(i, j)] =
                      (x0[IX(i, j)]
                      + a * (old_x[IX(i + 1, j)]
                      + old_x[IX(i - 1, j)]
                      + old_x[IX(i, j + 1)]
                      + old_x[IX(i, j - 1)]
                      )) * cRecip;
                }
            }

            set_bnd_compute(b, x);
        }
    }

    /// <summary>
    /// add force to the entirety of space, CPU mode
    /// </summary>
    private void add_external_force_cpu(float force, float[] d)
    {
        init_cpu();

        for (int x = 1; x < size - 1; x++)
        {
            for (int y = 1; y < size - 1; y++)
            {
                d[IX(x, y)] += force * dt;
            }
        }
    }

    /// <summary>
    /// regulate velocities to achieve zero divergence everywhere, CPU mode
    /// </summary>
    private void project_cpu(float[] velocX, float[] velocY, float[] p, float[] div)
    {
        init_cpu();

        for (int j = 1; j < size - 1; j++)
        {
            for (int i = 1; i < size - 1; i++)
            {
                div[IX(i, j)] = -0.5f * (
                  velocX[IX(i + 1, j)]
                  - velocX[IX(i - 1, j)]
                  + velocY[IX(i, j + 1)]
                  - velocY[IX(i, j - 1)]
                  ) / size;
                p[IX(i, j)] = 0;
            }
        }

        set_bnd_cpu(0, div);
        set_bnd_cpu(0, p);
        lin_solve_cpu(0, p, div, 1, 4);

        for (int j = 1; j < size - 1; j++)
        {
            for (int i = 1; i < size - 1; i++)
            {
                velocX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)]
                  - p[IX(i - 1, j)]) * size;
                velocY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)]
                  - p[IX(i, j - 1)]) * size;
            }
        }
        set_bnd_cpu(1, velocX);
        set_bnd_cpu(2, velocY);
    }

    /// <summary>
    /// flow given field in the direction of velocity fields, CPU mode
    /// </summary>
    private void advect_cpu(int b, float[] d, float[] d0, float[] velocX, float[] velocY)
    {
        init_cpu();

        float i0, i1, j0, j1;

        float dtx = dt * (size - 2);
        float dty = dt * (size - 2);

        float s0, s1, t0, t1;
        float tmp1, tmp2, x, y;

        float Nfloat = size;
        float ifloat, jfloat;
        int i, j;

        for (j = 1, jfloat = 1; j < size - 1; j++, jfloat++)
        {
            for (i = 1, ifloat = 1; i < size - 1; i++, ifloat++)
            {
                tmp1 = dtx * velocX[IX(i, j)];
                tmp2 = dty * velocY[IX(i, j)];

                x = ifloat - tmp1;
                y = jfloat - tmp2;

                if (x < 0.5f) x = 0.5f;
                if (x > Nfloat + 0.5f) x = Nfloat + 0.5f;
                i0 = Mathf.Floor(x);
                i1 = i0 + 1.0f;
                if (y < 0.5f) y = 0.5f;
                if (y > Nfloat + 0.5f) y = Nfloat + 0.5f;
                j0 = Mathf.Floor(y);
                j1 = j0 + 1.0f;

                s1 = x - i0;
                s0 = 1.0f - s1;
                t1 = y - j0;
                t0 = 1.0f - t1;

                int i0i = (int)i0;
                int i1i = (int)i1;
                int j0i = (int)j0;
                int j1i = (int)j1;


                d[IX(i, j)] =
                  s0 * (t0 * d0[IX(i0i, j0i)] + t1 * d0[IX(i0i, j1i)]) +
                  s1 * (t0 * d0[IX(i1i, j0i)] + t1 * d0[IX(i1i, j1i)]);
            }
        }

        set_bnd_compute(b, d);
    }

    #endregion

    #region GPU2D

    /// <summary>
    /// single iteration of simulation in GPU mode
    /// </summary>
    private void step_gpu()
    {
        _last_iteration_was_cpu = false;

        init_gpu();

        add_external_force_gpu2d(gravity.x * 0.0001f, Vx_rt);
        add_external_force_gpu2d(gravity.y * 0.0001f, Vy_rt);

        diffuse_gpu2d(1, Vx0_rt, Vx_rt);
        diffuse_gpu2d(2, Vy0_rt, Vy_rt);

        project_gpu2d(Vx0_rt, Vy0_rt, Vx_rt, Vy_rt);

        advect_gpu2d(1, Vx_rt, Vx0_rt, Vx0_rt, Vy0_rt);
        advect_gpu2d(2, Vy_rt, Vy0_rt, Vx0_rt, Vy0_rt);

        project_gpu2d(Vx_rt, Vy_rt, Vx0_rt, Vy0_rt);

        diffuse_gpu2d(0, s_rt, density_rt);
        advect_gpu2d(0, density_rt, s_rt, Vx_rt, Vy_rt);
    }

    /// <summary>
    /// add density to GPU buffer
    /// </summary>
    private void add_density_gpu(int x, int y, float amount)
    {
        init_gpu();
        _write_to_texture(density_rt, x, y, amount);
    }

    /// <summary>
    /// add velocity to GPU buffer
    /// </summary>
    private void add_velocity_gpu(int x, int y, float amount_x, float amount_y)
    {
        init_gpu();

        _write_to_texture(Vx_rt, x, y, amount_x);
        _write_to_texture(Vy_rt, x, y, amount_y);
    }

    private void _set_boundry_internal(int x, int y, float amount)
    {
        _write_to_texture(boundries_rt, x, y, amount);
    }

    private void set_boundry_gpu(int x0, int y0, int size_x, int size_y)
    {
        init_gpu();

        int new_boundry = 0;
        for (int x = x0; x < x0 + size_x; x++)
        {
            for (int y = y0; y < y0 + size_y; y++)
            {
                _set_boundry_internal(x, y, 1.0f);
                new_boundry++;
            }
        }
    }

    /// <summary>
    /// simulation behaviour at boundries, GPU mode
    /// </summary>
    private void set_bnd_gpu2d(int b, RenderTexture x)
    {
        init_gpu();

        int kernel_index = solver.FindKernel("set_bnd");
        solver.SetInt("size", size);
        solver.SetInt("b", b);
        solver.SetTexture(kernel_index, "x", x);
        solver.SetTexture(kernel_index, "boundries", boundries_rt);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);
    }

    /// <summary>
    /// simulate diffusion, GPU mode
    /// </summary>
    private void diffuse_gpu2d(int b, RenderTexture x, RenderTexture x0)
    {
        init_gpu();

        float a = dt * diff * (size - 2) * (size - 2);
        lin_solve_gpu2d(b, x, x0, a, 1 + 4 * a);
    }

    /// <summary>
    /// solve linear system, GPU mode
    /// </summary>
    private void lin_solve_gpu2d(int b, RenderTexture x, RenderTexture x0, float a, float c)
    {
        init_gpu();

        RenderTexture temp = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
        temp.enableRandomWrite = true;
        temp.filterMode = FilterMode.Bilinear;

        for (int i = 0; i < iter; i++)
        {
            if (linear_solver_method == GPULinearSolver.JACOBI)
            {
                int kernel_index = solver.FindKernel("copy_textures");
                solver.SetTexture(kernel_index, "copy_source_texture", x);
                solver.SetTexture(kernel_index, "copy_target_texture", temp);
                solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

                kernel_index = solver.FindKernel("lin_solve_jacobi");
                solver.SetInt("size", size);
                solver.SetInt("b", b);
                solver.SetFloat("a", a);
                solver.SetFloat("c", c);
                solver.SetTexture(kernel_index, "x", x);
                solver.SetTexture(kernel_index, "x0", x0);
                solver.SetTexture(kernel_index, "x_old", temp);
                solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);
            }
            if (linear_solver_method == GPULinearSolver.GAUSS_SEISEL)
            {
                int kernel_index = solver.FindKernel("lin_solve_gauss_seidel");
                solver.SetInt("size", size);
                solver.SetInt("b", b);
                solver.SetFloat("a", a);
                solver.SetFloat("c", c);
                solver.SetTexture(kernel_index, "x", x);
                solver.SetTexture(kernel_index, "x0", x0);
                solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);
            }

            set_bnd_gpu2d(b, x);
        }
    }

    /// <summary>
    /// add force to the entirety of space, GPU mode
    /// </summary>
    private void add_external_force_gpu2d(float force, RenderTexture d)
    {
        init_gpu();

        int kernel_index = solver.FindKernel("add_external_force");
        solver.SetInt("size", size);
        solver.SetFloat("force", force);
        solver.SetFloat("dt", dt);
        solver.SetTexture(kernel_index, "d", d);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        set_bnd_gpu2d(0, d);
    }

    /// <summary>
    /// regulate velocities to achieve zero divergence everywhere, GPU mode
    /// </summary>
    private void project_gpu2d(RenderTexture velocX, RenderTexture velocY, RenderTexture p, RenderTexture div)
    {
        init_gpu();

        int kernel_index = solver.FindKernel("project_1");
        solver.SetInt("size", size);
        solver.SetTexture(kernel_index, "p", p);
        solver.SetTexture(kernel_index, "div", div);
        solver.SetTexture(kernel_index, "velocity_x_1", velocX);
        solver.SetTexture(kernel_index, "velocity_y_1", velocY);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        set_bnd_gpu2d(0, div);
        set_bnd_gpu2d(0, p);
        lin_solve_gpu2d(0, p, div, 1, 4);

        kernel_index = solver.FindKernel("project_2");
        solver.SetInt("size", size);
        solver.SetTexture(kernel_index, "p", p);
        solver.SetTexture(kernel_index, "div", div);
        solver.SetTexture(kernel_index, "velocity_x_1", velocX);
        solver.SetTexture(kernel_index, "velocity_y_1", velocY);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        set_bnd_gpu2d(1, velocX);
        set_bnd_gpu2d(2, velocY);
    }

    /// <summary>
    /// flow given field in the direction of velocity fields, GPU mode
    /// </summary>
    private void advect_gpu2d(int b, RenderTexture d, RenderTexture d0, RenderTexture velocX, RenderTexture velocY)
    {
        init_gpu();

        RenderTexture vx_copy = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
        vx_copy.enableRandomWrite = true;
        vx_copy.filterMode = FilterMode.Bilinear;

        int kernel_index = solver.FindKernel("copy_textures");
        solver.SetTexture(kernel_index, "copy_source_texture", velocX);
        solver.SetTexture(kernel_index, "copy_target_texture", vx_copy);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        RenderTexture vy_copy = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
        vy_copy.enableRandomWrite = true;
        vy_copy.filterMode = FilterMode.Bilinear;

        kernel_index = solver.FindKernel("copy_textures");
        solver.SetTexture(kernel_index, "copy_source_texture", velocY);
        solver.SetTexture(kernel_index, "copy_target_texture", vy_copy);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        kernel_index = solver.FindKernel("advect");
        solver.SetInt("size", size);
        solver.SetInt("b", b);
        solver.SetFloat("dt", dt);
        solver.SetTexture(kernel_index, "d", d);
        solver.SetTexture(kernel_index, "d0", d0);
        solver.SetTexture(kernel_index, "velocity_x_1", vx_copy);
        solver.SetTexture(kernel_index, "velocity_y_1", vy_copy);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size / 8.0f), Mathf.CeilToInt(size / 8.0f), 1);

        set_bnd_gpu2d(b, d);
    }

    #endregion

    #region GPU3D

    // privates

    private float step_total_cost_3d;

    private float write_to_texture_total_cost_3d;
    private int write_to_texture_call_count_3d;

    private float set_bnd_total_cost_3d;
    private int set_bnd_call_count_3d;

    private float lin_solve_total_cost_3d;
    private int lin_solve_call_count_3d;

    private float project_total_cost_3d;
    private int project_call_count_3d;

    private float advect_total_cost_3d;
    private int advect_call_count_3d;

    private float copy_textures_total_cost_3d;
    private int copy_textures_call_count_3d;

    private int print_performance_every = 100;
    private int performance_printer_counter = 0;
    public bool print_performance;
    private void clear_performance_parameters()
    {
        if (--performance_printer_counter <= 0)
        {
            if (print_performance)
            {
                string performance_string = "step_total_cost_3d " + step_total_cost_3d + " ms" + "\n" +
                                            "write_to_texture_total_cost_3d " + write_to_texture_total_cost_3d + " ms" + "\n" +
                                            "write_to_texture_call_count_3d " + write_to_texture_call_count_3d + " call" + "\n" +
                                            "set_bnd_total_cost_3d " + set_bnd_total_cost_3d + " ms" + "\n" +
                                            "set_bnd_call_count_3d " + set_bnd_call_count_3d + " call" + "\n" +
                                            "lin_solve_total_cost_3d " + lin_solve_total_cost_3d + " ms" + "\n" +
                                            "lin_solve_call_count_3d " + lin_solve_call_count_3d + " call" + "\n" +
                                            "project_total_cost_3d " + project_total_cost_3d + " ms" + "\n" +
                                            "project_call_count_3d " + project_call_count_3d + " call" + "\n" +
                                            "advect_total_cost_3d " + advect_total_cost_3d + " ms" + "\n" +
                                            "advect_call_count_3d " + advect_call_count_3d + " call" + "\n" +
                                            "copy_textures_total_cost_3d " + copy_textures_total_cost_3d + " ms" + "\n" +
                                            "copy_textures_call_count_3d " + copy_textures_call_count_3d + " call" + "\n";

                Debug.Log(performance_string);
            }

            step_total_cost_3d = 0;
            write_to_texture_total_cost_3d = 0;
            write_to_texture_call_count_3d = 0;
            set_bnd_total_cost_3d = 0;
            set_bnd_call_count_3d = 0;
            lin_solve_total_cost_3d = 0;
            lin_solve_call_count_3d = 0;
            project_total_cost_3d = 0;
            project_call_count_3d = 0;
            advect_total_cost_3d = 0;
            advect_call_count_3d = 0;
            copy_textures_total_cost_3d = 0;
            copy_textures_call_count_3d = 0;

            performance_printer_counter = print_performance_every;
        }
    }

    /// <summary>
    /// single iteration of simulation in GPU3D mode
    /// </summary>
    private void step_gpu_3d()
    {
        clear_performance_parameters();
        float step_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();

        _last_iteration_was_cpu = false;

        add_external_force_gpu3d(gravity.x * 0.0001f, Vx_rt);
        add_external_force_gpu3d(gravity.y * 0.0001f, Vy_rt);
        add_external_force_gpu3d(gravity.z * 0.0001f, Vz_rt);

        diffuse_gpu3d(1, Vx0_rt, Vx_rt);
        diffuse_gpu3d(2, Vy0_rt, Vy_rt);
        diffuse_gpu3d(3, Vz0_rt, Vz_rt);

        project_gpu3d(Vx0_rt, Vy0_rt, Vz0_rt, Vx_rt, Vy_rt);

        advect_gpu3d(1, Vx_rt, Vx0_rt, Vx0_rt, Vy0_rt, Vz0_rt);
        advect_gpu3d(2, Vy_rt, Vy0_rt, Vx0_rt, Vy0_rt, Vz0_rt);
        advect_gpu3d(3, Vz_rt, Vz0_rt, Vx0_rt, Vy0_rt, Vz0_rt);

        project_gpu3d(Vx_rt, Vy_rt, Vz_rt, Vx0_rt, Vy0_rt);

        diffuse_gpu3d(0, s_rt, density_rt);
        advect_gpu3d(0, density_rt, s_rt, Vx_rt, Vy_rt, Vz_rt);

        step_total_cost_3d += Time.realtimeSinceStartup * 1000 - step_begin_time;
    }

    /// <summary>
    /// add density to GPU buffer
    /// </summary>
    private void add_density_gpu3d(int x, int y, int z, float amount)
    {
        init_gpu();

        _write_to_texture_3d(density_rt, x, y, z, amount);
    }

    /// <summary>
    /// add velocity to GPU buffer
    /// </summary>
    private void add_velocity_gpu3d(int x, int y, int z, float amount_x, float amount_y, float amount_z)
    {
        init_gpu();

        _write_to_texture_3d(Vx_rt, x, y, z, amount_x);
        _write_to_texture_3d(Vy_rt, x, y, z, amount_y);
        _write_to_texture_3d(Vz_rt, x, y, z, amount_z);
    }

    private void _set_boundry_internal3d(int x, int y, int z, float amount)
    {
        _write_to_texture_3d(boundries_rt, x, y, z, amount);
    }

    private void set_boundry_gpu3d(int x0, int y0, int z0, int size_x, int size_y, int size_z)
    {
        init_gpu();

        int new_boundry = 0;
        for (int x = x0; x < x0 + size_x; x++)
        {
            for (int y = y0; y < y0 + size_y; y++)
            {
                for (int z = z0; z < z0 + size_z; z++)
                {
                    _set_boundry_internal3d(x, y, z, 1.0f);
                    new_boundry++;
                }
            }
        }
    }

    /// <summary>
    /// simulation behaviour at boundries, GPU mode
    /// </summary>
    private void set_bnd_gpu3d(int b, RenderTexture x)
    {
        float set_bnd_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();

        int kernel_index = solver3D.FindKernel("set_bnd_3d");
        solver3D.SetInt("size", size);
        solver3D.SetInt("b", b);
        solver3D.SetTexture(kernel_index, "x", x);
        solver3D.SetTexture(kernel_index, "boundries", boundries_rt);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

        set_bnd_total_cost_3d += Time.realtimeSinceStartup * 1000 - set_bnd_begin_time;
        set_bnd_call_count_3d++;
    }

    /// <summary>
    /// simulate diffusion, GPU mode
    /// </summary>
    private void diffuse_gpu3d(int b, RenderTexture x, RenderTexture x0)
    {
        init_gpu();

        float a = dt * diff * (size - 2) * (size - 2);
        lin_solve_gpu3d(b, x, x0, a, 1 + 6 * a);
    }

    // second copy of a texture used for jacobi iterations
    private RenderTexture temp_linear_solve_jacobi;

    /// <summary>
    /// solve linear system, GPU mode
    /// </summary>
    private void lin_solve_gpu3d(int b, RenderTexture x, RenderTexture x0, float a, float c)
    {
        float lin_solve_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();
        if (linear_solver_method == GPULinearSolver.JACOBI)
        {
            if (temp_linear_solve_jacobi == null || temp_linear_solve_jacobi.width != size)
            {
                temp_linear_solve_jacobi = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
                temp_linear_solve_jacobi.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                temp_linear_solve_jacobi.volumeDepth = size;
                temp_linear_solve_jacobi.enableRandomWrite = true;
                temp_linear_solve_jacobi.filterMode = FilterMode.Bilinear;
            }

            int copy_kernel_index = solver3D.FindKernel("copy_textures_3d");
            solver3D.SetTexture(copy_kernel_index, "copy_source_texture", x);
            solver3D.SetTexture(copy_kernel_index, "copy_target_texture", temp_linear_solve_jacobi);

            int lin_solve_kernel_index = solver3D.FindKernel("lin_solve_jacobi_3d");
            solver3D.SetInt("size", size);
            solver3D.SetInt("b", b);
            solver3D.SetFloat("a", a);
            solver3D.SetFloat("c", c);
            solver3D.SetTexture(lin_solve_kernel_index, "x", x);
            solver3D.SetTexture(lin_solve_kernel_index, "x0", x0);
            solver3D.SetTexture(lin_solve_kernel_index, "x_old", temp_linear_solve_jacobi);

            for (int i = 0; i < iter; i++)
            {
                float copy_textures_begin_time = Time.realtimeSinceStartup * 1000;

                solver3D.Dispatch(copy_kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

                copy_textures_total_cost_3d += Time.realtimeSinceStartup * 1000 - copy_textures_begin_time;
                copy_textures_call_count_3d++;

                solver3D.Dispatch(lin_solve_kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));
            }
        }

        if (linear_solver_method == GPULinearSolver.GAUSS_SEISEL)
        {
            int kernel_index = solver3D.FindKernel("lin_solve_gauss_seidel_3d");
            solver3D.SetInt("size", size);
            solver3D.SetInt("b", b);
            solver3D.SetFloat("a", a);
            solver3D.SetFloat("c", c);
            solver3D.SetTexture(kernel_index, "x", x);
            solver3D.SetTexture(kernel_index, "x0", x0);
            for (int i = 0; i < iter; i++)
                solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));
        }

        set_bnd_gpu3d(b, x);

        lin_solve_total_cost_3d += Time.realtimeSinceStartup * 1000 - lin_solve_begin_time;
        lin_solve_call_count_3d++;
    }

    /// <summary>
    /// add force to the entirety of space, GPU mode
    /// </summary>
    private void add_external_force_gpu3d(float force, RenderTexture d)
    {
        init_gpu();

        int kernel_index = solver3D.FindKernel("add_external_force_3d");
        solver3D.SetInt("size", size);
        solver3D.SetFloat("force", force);
        solver3D.SetFloat("dt", dt);
        solver3D.SetTexture(kernel_index, "d", d);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

        set_bnd_gpu3d(0, d);
    }

    /// <summary>
    /// regulate velocities to achieve zero divergence everywhere, GPU mode
    /// </summary>
    private void project_gpu3d(RenderTexture velocX, RenderTexture velocY, RenderTexture velocZ, RenderTexture p, RenderTexture div)
    {
        float project_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();

        int kernel_index = solver3D.FindKernel("project_1_3d");
        solver3D.SetInt("size", size);
        solver3D.SetTexture(kernel_index, "p", p);
        solver3D.SetTexture(kernel_index, "div", div);
        solver3D.SetTexture(kernel_index, "velocity_x_1", velocX);
        solver3D.SetTexture(kernel_index, "velocity_y_1", velocY);
        solver3D.SetTexture(kernel_index, "velocity_z_1", velocZ);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

        set_bnd_gpu3d(0, div);
        set_bnd_gpu3d(0, p);
        lin_solve_gpu3d(0, p, div, 1, 6);

        kernel_index = solver3D.FindKernel("project_2_3d");
        solver3D.SetInt("size", size);
        solver3D.SetTexture(kernel_index, "p", p);
        solver3D.SetTexture(kernel_index, "div", div);
        solver3D.SetTexture(kernel_index, "velocity_x_1", velocX);
        solver3D.SetTexture(kernel_index, "velocity_y_1", velocY);
        solver3D.SetTexture(kernel_index, "velocity_z_1", velocZ);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

        set_bnd_gpu3d(1, velocX);
        set_bnd_gpu3d(2, velocY);
        set_bnd_gpu3d(3, velocZ);

        project_total_cost_3d += Time.realtimeSinceStartup * 1000 - project_begin_time;
        project_call_count_3d++;
    }

    // second copies of velocities for advect
    private RenderTexture temp_advect_vx;
    private RenderTexture temp_advect_vy;
    private RenderTexture temp_advect_vz;

    /// <summary>
    /// flow given field in the direction of velocity fields, GPU mode
    /// </summary>
    private void advect_gpu3d(int b, RenderTexture d, RenderTexture d0, RenderTexture velocX, RenderTexture velocY, RenderTexture velocZ)
    {
        float advect_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();

        bool should_initialize_temps = temp_advect_vx == null || temp_advect_vy == null || temp_advect_vz == null;
        should_initialize_temps = should_initialize_temps || temp_advect_vx.width != size || temp_advect_vy.width != size || temp_advect_vz.width != size;

        int kernel_index;

        if (should_initialize_temps)
        {
            temp_advect_vx = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            temp_advect_vx.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            temp_advect_vx.volumeDepth = size;
            temp_advect_vx.enableRandomWrite = true;
            temp_advect_vx.filterMode = FilterMode.Bilinear;

            temp_advect_vy = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            temp_advect_vy.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            temp_advect_vy.volumeDepth = size;
            temp_advect_vy.enableRandomWrite = true;
            temp_advect_vy.filterMode = FilterMode.Bilinear;

            temp_advect_vz = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat, 0);
            temp_advect_vz.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            temp_advect_vz.volumeDepth = size;
            temp_advect_vz.enableRandomWrite = true;
            temp_advect_vz.filterMode = FilterMode.Bilinear;
        }

        float texture_copies_begin_time = Time.realtimeSinceStartup * 1000;

        kernel_index = solver3D.FindKernel("copy_textures_3d");
        solver3D.SetTexture(kernel_index, "copy_source_texture", velocX);
        solver3D.SetTexture(kernel_index, "copy_target_texture", temp_advect_vx);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

        kernel_index = solver3D.FindKernel("copy_textures_3d");
        solver3D.SetTexture(kernel_index, "copy_source_texture", velocY);
        solver3D.SetTexture(kernel_index, "copy_target_texture", temp_advect_vy);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

        kernel_index = solver3D.FindKernel("copy_textures_3d");
        solver3D.SetTexture(kernel_index, "copy_source_texture", velocZ);
        solver3D.SetTexture(kernel_index, "copy_target_texture", temp_advect_vz);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

        copy_textures_total_cost_3d += Time.realtimeSinceStartup * 1000 - texture_copies_begin_time;
        copy_textures_call_count_3d += 3;

        kernel_index = solver3D.FindKernel("advect_3d");
        solver3D.SetInt("size", size);
        solver3D.SetInt("b", b);
        solver3D.SetFloat("dt", dt);
        solver3D.SetTexture(kernel_index, "d", d);
        solver3D.SetTexture(kernel_index, "d0", d0);
        solver3D.SetTexture(kernel_index, "velocity_x_1", temp_advect_vx);
        solver3D.SetTexture(kernel_index, "velocity_y_1", temp_advect_vy);
        solver3D.SetTexture(kernel_index, "velocity_z_1", temp_advect_vz);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), Mathf.CeilToInt(size / solver3D_kernel_size.z));

        set_bnd_gpu3d(b, d);

        advect_total_cost_3d += Time.realtimeSinceStartup * 1000 - advect_begin_time;
        advect_call_count_3d++;
    }

    #endregion

    /// <summary>
    /// set given rectangular area as boundry to current mode's buffer
    /// </summary>
    public void set_boundry(int x0, int y0, int size_x, int size_y)
    {
        if (mode == ComputationMode.GPU3D) return;
        if (mode == ComputationMode.GPU2D) set_boundry_gpu(x0, y0, size_x, size_y);
        if (mode == ComputationMode.CPU2D || mode == ComputationMode.COMPUTE_DEBUG2D) set_boundry_cpu(x0, y0, size_x, size_y);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_density(int x, int y, float amount)
    {
        if (mode == ComputationMode.GPU3D) return;
        if (mode == ComputationMode.GPU2D) add_density_gpu(x, y, amount);
        if (mode == ComputationMode.CPU2D || mode == ComputationMode.COMPUTE_DEBUG2D) add_density_cpu(x, y, amount);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_velocity(int x, int y, float amount_x, float amount_y)
    {
        if (mode == ComputationMode.GPU3D) return;
        if (mode == ComputationMode.GPU2D) add_velocity_gpu(x, y, amount_x, amount_y);
        if (mode == ComputationMode.CPU2D || mode == ComputationMode.COMPUTE_DEBUG2D) add_velocity_cpu(x, y, amount_x, amount_y);
    }

    /// <summary>
    /// set given rectangular area as boundry to current mode's buffer
    /// </summary>
    public void set_boundry(int x0, int y0, int z0, int size_x, int size_y, int size_z)
    {
        if (mode == ComputationMode.GPU2D) set_boundry_gpu(x0, y0, size_x, size_y);
        if (mode == ComputationMode.GPU3D) set_boundry_gpu3d(x0, y0, z0, size_x, size_y, size_z);
        if (mode == ComputationMode.CPU2D || mode == ComputationMode.COMPUTE_DEBUG2D) set_boundry_cpu(x0, y0, size_x, size_y);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_density(int x, int y, int z, float amount)
    {
        if (mode == ComputationMode.GPU2D) add_density_gpu(x, y, amount);
        if (mode == ComputationMode.GPU3D) add_density_gpu3d(x, y, z, amount);
        if (mode == ComputationMode.CPU2D || mode == ComputationMode.COMPUTE_DEBUG2D) add_density_cpu(x, y, amount);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_velocity(int x, int y, int z, float amount_x, float amount_y, float amount_z)
    {
        if (mode == ComputationMode.GPU2D) add_velocity_gpu(x, y, amount_x, amount_y);
        if (mode == ComputationMode.GPU3D) add_velocity_gpu3d(x, y, z, amount_x, amount_y, amount_z);
        if (mode == ComputationMode.CPU2D || mode == ComputationMode.COMPUTE_DEBUG2D) add_velocity_cpu(x, y, amount_x, amount_y);
    }

    public void add_density_worldcoord(Vector3 world_coord, float amount)
    {
        Vector3 pixel_coord = (world_coord - transform.position + transform.lossyScale / 2) / transform.lossyScale.x * size;
        add_density((int)pixel_coord.x, (int)pixel_coord.y, (int)pixel_coord.z, amount);
    }

    public void add_velocity_worldcoord(Vector3 world_coord, Vector3 velocity)
    {
        Vector3 pixel_coord = (world_coord - transform.position + transform.lossyScale / 2) / transform.lossyScale.x * size;
        add_velocity((int)pixel_coord.x, (int)pixel_coord.y, (int)pixel_coord.z, velocity.x, velocity.y, velocity.z);
    }

    private float last_measured_second;
    private int step_per_second_temp;
    public int step_per_second;

    /// <summary>
    /// iterate time forward
    /// </summary>
    public void step(float deltatime_ms)
    {
        // count number of steps per second for performance profiling
        if ((int)(last_measured_second) != (int)(Time.realtimeSinceStartup))
        {
            last_measured_second = Time.realtimeSinceStartup;
            step_per_second = step_per_second_temp;
            step_per_second_temp = 0;
        }

        carryover_time_ms += deltatime_ms;
        if (carryover_time_ms > target_frametime_ms)
        {
            step_per_second_temp++;
            carryover_time_ms -= target_frametime_ms;
            if (mode == ComputationMode.GPU2D && _last_iteration_was_cpu) transfer_data_from_cpu_to_gpu();
            if ((mode == ComputationMode.CPU2D || mode == ComputationMode.COMPUTE_DEBUG2D) && !_last_iteration_was_cpu) transfer_data_from_gpu_to_cpu();

            if (mode == ComputationMode.GPU3D) step_gpu_3d();
            else if (mode == ComputationMode.GPU2D) step_gpu();
            else if (mode == ComputationMode.CPU2D) step_cpu();
            else if (mode == ComputationMode.COMPUTE_DEBUG2D) step_compute();
        }
    }

    public int simulation_resolution;
    public int simulation_iteration_per_step;
    public float diffusion;

    public int wall_thickness;
    void Start()
    {
        init_parameters(simulation_resolution, simulation_iteration_per_step, 0.1f, 0.0001f * diffusion, 0.1f);

        set_boundry(0, 0, 0, wall_thickness, size, size);
        set_boundry(0, 0, 0, size, wall_thickness, size);
        set_boundry(0, 0, 0, size, size, wall_thickness);

        set_boundry(size - wall_thickness, 0, 0, wall_thickness, size, size);
        set_boundry(0, size - wall_thickness, 0, size, wall_thickness, size);
        set_boundry(0, 0, size - wall_thickness, size, size, wall_thickness);

        //set_boundry(0, 0, 0, size, wall_thickness, size);
        //set_boundry(0, 0, 0, size, size, wall_thickness);


        //set_boundry(0, size - wall_thickness, 0, size, wall_thickness, size);
        //set_boundry(0, 0, 0, wall_thickness, size, size);
        //set_boundry(size - wall_thickness, 0, wall_thickness, size);


        //set_boundry(30, 30, 16, 16);
        //
        //for (int i = 0; i < size; i++)
        //    for (int j = 0; j < size; j++)
        //        if (Mathf.Pow(i - 10, 2) + Mathf.Pow(j - 30, 2) < 8 * 8)
        //            set_boundry(i, j, 1, 1);
    }

    [Range(1, 10)]
    public int mouse_add_size_density;
    [Range(1, 10)]
    public int mouse_add_size_velocity;

    [Range(0, 16)]
    public float mouse_add_density;
    public Vector3 mouse_add_velocity;

    void FixedUpdate()
    {
        init_parameters(simulation_resolution, simulation_iteration_per_step, 0.1f, 0.0001f * diffusion, 0.1f);

        if (Input.GetKey(KeyCode.Mouse0))
        {
            int cx = (int)(Input.mousePosition.x / Screen.width * size);
            int cy = (int)(Input.mousePosition.y / Screen.height * size);

            for (int i = 0; i < mouse_add_size_density; i++)
                for (int j = 0; j < mouse_add_size_density; j++)
                    add_density(cx + i, cy + j, size / 4 * 3, mouse_add_density);

            for (int i = 0; i < mouse_add_size_velocity; i++)
                for (int j = 0; j < mouse_add_size_velocity; j++)
                    add_velocity(cx + i, cy + j, size / 4 * 3, mouse_add_velocity.x, mouse_add_velocity.y, 0);
        }

        if (Input.GetKey(KeyCode.Space))
        {
            for (int i = 0; i < mouse_add_size_density; i++)
                for (int j = 0; j < mouse_add_size_density; j++)
                    for (int k = 0; k < mouse_add_size_density; k++)
                        add_density(size / 2 + i, size / 4 + j, size / 2 + k, mouse_add_density);

            for (int i = 0; i < mouse_add_size_velocity; i++)
                for (int j = 0; j < mouse_add_size_velocity; j++)
                    for (int k = 0; k < mouse_add_size_velocity; k++)
                        add_velocity(size / 2 + i, size / 4 + j, size / 2 + k, (Random.value - 0.5f) * mouse_add_velocity.x, mouse_add_velocity.y, (Random.value - 0.5f) * mouse_add_velocity.z);
        }


    }


    public bool iterate_time;
    private void Update()
    {
        if (iterate_time)
            step(Time.deltaTime * 1000);
    }

    bool _first_draw_in_gpu_mode;
    bool _first_draw_in_cpu_mode;

    // texture to display cpu mode buffers
    Texture2D texture;
    RenderTexture slice_of_3D;
    public int displaying_slice_index;
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (_last_iteration_was_cpu)
        {

            if (_first_draw_in_cpu_mode)
            {
                init_cpu();
                texture = new Texture2D(size, size, TextureFormat.RFloat, false);
                texture.filterMode = FilterMode.Bilinear;
            }
            _first_draw_in_cpu_mode = false;

            Color[] texture_data = new Color[size * size];

            for (int x = 0; x < size; x++)
            {
                for (int y = 0; y < size; y++)
                {
                    texture_data[y * size + x].r = density[IX(x, y)];
                    if (boundries[IX(x, y)]) texture_data[y * size + x].r = 0;
                }
            }

            texture.SetPixels(texture_data);
            texture.Apply();

            Graphics.Blit(texture, (RenderTexture)null);
        }

        if (!_last_iteration_was_cpu)
        {

            if (_first_draw_in_gpu_mode)
            {
                init_gpu();
                slice_of_3D = new RenderTexture(size, size, 0);
                slice_of_3D.filterMode = FilterMode.Bilinear;
                slice_of_3D.enableRandomWrite = true;
            }
            _first_draw_in_gpu_mode = false;

            if (mode == ComputationMode.GPU2D)
                Graphics.Blit(density_rt, (RenderTexture)null);
            if (mode == ComputationMode.GPU3D)
            {
                int kernel_index = solver3D.FindKernel("get_3d_texture_slice");
                solver3D.SetInt("slice_z_index", displaying_slice_index);
                solver3D.SetTexture(kernel_index, "copy_source_texture", density_rt);
                solver3D.SetTexture(kernel_index, "slice_target", slice_of_3D);
                solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size / solver3D_kernel_size.x), Mathf.CeilToInt(size / solver3D_kernel_size.y), 1);
                Graphics.Blit(slice_of_3D, (RenderTexture)null);
            }

        }


    }
}
