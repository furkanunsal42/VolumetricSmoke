using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleNavierStokesSolver : MonoBehaviour    // referance paper : https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
{
    public static HashSet<SimpleNavierStokesSolver> all_instances = new HashSet<SimpleNavierStokesSolver>();

    public void OnEnable()
    {
        all_instances.Add(this);
    }

    public void OnDisable()
    {
        all_instances.Remove(this);
    }

    // simulation parameters
    private Vector3Int size;
    private int iter;

    public Vector3 gravity = new Vector3(0, -9.8f, 0);

    private float dt;
    private float diff;
    private float visc;

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
    private bool _gpu_buffers_initialized = false;
    private bool _gpu_buffers_sized_for_3D = false;

    public float target_frametime_ms;
    private float carryover_time_ms;

    public enum ComputationMode
    {
        GPU3D,
        GPU2D,
    }

    [SerializeField] private ComputationMode mode;

    public enum GPULinearSolver
    {
        JACOBI,
        GAUSS_SEISEL
    }

    public GPULinearSolver linear_solver_method;

    public SimpleNavierStokesSolver(Vector3Int size, float dt, int iterations, float diffusion, float viscosity)
    {
        init_parameters(size, iterations, dt, diffusion, viscosity);
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
    public void init_parameters(Vector3Int size, int iterations, float dt, float diffusion, float viscosity)
    {
        solver3D_kernel_size = new Vector3(2, 2, 2);

        if (size != this.size)
        {
            _gpu_buffers_initialized = false;
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
            
            free_vram();

            this.s_rt =         new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.density_rt =   new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vx_rt =        new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vy_rt =        new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vz_rt =        new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vx0_rt =       new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vy0_rt =       new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vz0_rt =       new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.boundries_rt = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);

            this.s_rt.dimension         = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.density_rt.dimension   = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vx_rt.dimension        = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vy_rt.dimension        = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vz_rt.dimension        = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vx0_rt.dimension       = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vy0_rt.dimension       = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.Vz0_rt.dimension       = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.boundries_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;

            this.s_rt.volumeDepth           = size.z;
            this.density_rt.volumeDepth     = size.z;
            this.Vx_rt.volumeDepth          = size.z;
            this.Vy_rt.volumeDepth          = size.z;
            this.Vz_rt.volumeDepth          = size.z;
            this.Vx0_rt.volumeDepth         = size.z;
            this.Vy0_rt.volumeDepth         = size.z;
            this.Vz0_rt.volumeDepth         = size.z;
            this.boundries_rt.volumeDepth   = size.z;

            this.s_rt.wrapMode          = TextureWrapMode.Clamp;
            this.density_rt.wrapMode    = TextureWrapMode.Clamp;
            this.Vx_rt.wrapMode         = TextureWrapMode.Clamp;
            this.Vy_rt.wrapMode         = TextureWrapMode.Clamp;
            this.Vz_rt.wrapMode         = TextureWrapMode.Clamp;
            this.Vx0_rt.wrapMode        = TextureWrapMode.Clamp;
            this.Vy0_rt.wrapMode        = TextureWrapMode.Clamp;
            this.Vz0_rt.wrapMode        = TextureWrapMode.Clamp;
            this.boundries_rt.wrapMode  = TextureWrapMode.Clamp;

            this.s_rt.enableRandomWrite         = true;
            this.density_rt.enableRandomWrite   = true;
            this.Vx_rt.enableRandomWrite        = true;
            this.Vy_rt.enableRandomWrite        = true;
            this.Vz_rt.enableRandomWrite        = true;
            this.Vx0_rt.enableRandomWrite       = true;
            this.Vy0_rt.enableRandomWrite       = true;
            this.Vz0_rt.enableRandomWrite       = true;
            this.boundries_rt.enableRandomWrite = true;

            this.s_rt.filterMode            = FilterMode.Bilinear;
            this.density_rt.filterMode      = FilterMode.Bilinear;
            this.Vx_rt.filterMode           = FilterMode.Bilinear;
            this.Vy_rt.filterMode           = FilterMode.Bilinear;
            this.Vz_rt.filterMode           = FilterMode.Bilinear;
            this.Vx0_rt.filterMode          = FilterMode.Bilinear;
            this.Vy0_rt.filterMode          = FilterMode.Bilinear;
            this.Vz0_rt.filterMode          = FilterMode.Bilinear;
            this.boundries_rt.filterMode    = FilterMode.Bilinear;
        }
        else
        {
            _gpu_buffers_sized_for_3D = false;

            Debug.Log("2d gpu init");

            if(this.s_rt        ) this.s_rt.Release();        
            if(this.density_rt  ) this.density_rt.Release();
            if (this.Vx_rt      ) this.Vx_rt.Release();   
            if(this.Vy_rt       ) this.Vy_rt.Release();      
            if(this.Vz_rt       ) this.Vz_rt.Release();     
            if(this.Vx0_rt      ) this.Vx0_rt.Release();
            if(this.Vy0_rt      ) this.Vy0_rt.Release();
            if(this.Vz0_rt      ) this.Vz0_rt.Release();
            if(this.boundries_rt) this.boundries_rt.Release();

            this.s_rt           = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.density_rt     = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vx_rt          = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vy_rt          = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vz_rt          = null;
            this.Vx0_rt         = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vy0_rt         = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.Vz0_rt         = null;
            this.boundries_rt   = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            
            this.s_rt.wrapMode          = TextureWrapMode.Clamp;
            this.density_rt.wrapMode    = TextureWrapMode.Clamp;
            this.Vx_rt.wrapMode         = TextureWrapMode.Clamp;
            this.Vy_rt.wrapMode         = TextureWrapMode.Clamp;
            this.Vx0_rt.wrapMode        = TextureWrapMode.Clamp;
            this.Vy0_rt.wrapMode        = TextureWrapMode.Clamp;
            this.boundries_rt.wrapMode  = TextureWrapMode.Clamp;

            this.s_rt.enableRandomWrite         = true;
            this.density_rt.enableRandomWrite   = true;
            this.Vx_rt.enableRandomWrite        = true;
            this.Vy_rt.enableRandomWrite        = true;
            this.Vx0_rt.enableRandomWrite       = true;
            this.Vy0_rt.enableRandomWrite       = true;
            this.boundries_rt.enableRandomWrite = true;

            this.s_rt.filterMode            = FilterMode.Bilinear;
            this.density_rt.filterMode      = FilterMode.Bilinear;
            this.Vx_rt.filterMode           = FilterMode.Bilinear;
            this.Vy_rt.filterMode           = FilterMode.Bilinear;
            this.Vx0_rt.filterMode          = FilterMode.Bilinear;
            this.Vy0_rt.filterMode          = FilterMode.Bilinear;
            this.boundries_rt.filterMode    = FilterMode.Bilinear;
        }

    }

    #region UTILS
    /// <summary>
    /// add the given amount to a single index of a GPU buffer
    /// </summary>
    private void _write_to_texture(RenderTexture target, int x, int y, float amount, bool block_writes_to_edges = false)
    {
        int upper_x_limit = block_writes_to_edges ? size.x - 2 : size.x - 1;
        int upper_y_limit = block_writes_to_edges ? size.y - 2 : size.y - 1;
        int lower_x_limit = block_writes_to_edges ? 1 : 0;
        int lower_y_limit = block_writes_to_edges ? 1 : 0;

        if (x > upper_x_limit || x < lower_x_limit) return;
        if (y > upper_y_limit || y < lower_y_limit) return;

        int kernel_index = solver.FindKernel("write_to_texture");
        solver.SetVector("write_position", new Vector2(x, y));
        solver.SetFloat("value", amount);
        solver.SetTexture(kernel_index, "write_target", target);
        solver.Dispatch(kernel_index, 1, 1, 1);
    }

    /// <summary>
    /// add the given amount to a single index of a GPU3D buffer
    /// </summary>
    private void _write_to_texture_3d(RenderTexture target, int x, int y, int z, float amount, bool block_writes_to_edges = false)
    {
        int upper_x_limit = block_writes_to_edges ? size.x - 2 : size.x - 1;
        int upper_y_limit = block_writes_to_edges ? size.y - 2 : size.y - 1;
        int upper_z_limit = block_writes_to_edges ? size.z - 2 : size.z - 1;
        int lower_x_limit = block_writes_to_edges ? 1 : 0;
        int lower_y_limit = block_writes_to_edges ? 1 : 0;
        int lower_z_limit = block_writes_to_edges ? 1 : 0;

        if (x > upper_x_limit || x < lower_x_limit) return;
        if (y > upper_y_limit || y < lower_y_limit) return;
        if (z > upper_z_limit || z < lower_z_limit) return;

        float write_to_texture_begin_time = Time.realtimeSinceStartup * 1000;

        int kernel_index = solver3D.FindKernel("write_to_texture_3d");
        solver3D.SetVector("write_position", new Vector3(x, y, z));
        solver3D.SetFloat("value", amount);
        solver3D.SetTexture(kernel_index, "write_target", target);
        solver3D.Dispatch(kernel_index, 1, 1, 1);

        write_to_texture_total_cost_3d += Time.realtimeSinceStartup * 1000 - write_to_texture_begin_time;
        write_to_texture_call_count_3d++;
    }

    #endregion

    #region GPU2D

    /// <summary>
    /// single iteration of simulation in GPU mode
    /// </summary>
    private void step_gpu()
    {
        init_gpu();

        add_external_force_gpu2d(gravity.x * 0.0001f, Vx_rt);
        add_external_force_gpu2d(gravity.y * 0.0001f, Vy_rt);

        //diffuse_gpu2d(1, Vx0_rt, Vx_rt);
        //diffuse_gpu2d(2, Vy0_rt, Vy_rt);
        //diffuse_gpu2d(0, s_rt, density_rt);

        //project_gpu2d(Vx0_rt, Vy0_rt, Vx_rt, Vy_rt);
        //project_gpu2d(Vx_rt, Vy_rt, Vx0_rt, Vy0_rt);

        advect_gpu2d(1, Vx_rt, Vx0_rt, Vx0_rt, Vy0_rt);
        advect_gpu2d(2, Vy_rt, Vy0_rt, Vx0_rt, Vy0_rt);
        advect_gpu2d(0, density_rt, s_rt, Vx_rt, Vy_rt);

        swap_textures_gpu2d(Vx_rt, Vx0_rt);
        swap_textures_gpu2d(Vy_rt, Vy0_rt);
        swap_textures_gpu2d(density_rt, s_rt);
    }

    /// <summary>
    /// add density to GPU buffer
    /// </summary>
    private void add_density_gpu(int x, int y, float amount)
    {
        init_gpu();
        _write_to_texture(density_rt, x, y, amount, true);
    }

    /// <summary>
    /// add velocity to GPU buffer
    /// </summary>
    private void add_velocity_gpu(int x, int y, float amount_x, float amount_y)
    {
        init_gpu();

        _write_to_texture(Vx_rt, x, y, amount_x, true);
        _write_to_texture(Vy_rt, x, y, amount_y, true);
    }

    private void _set_boundry_internal(int x, int y, float amount)
    {
        _write_to_texture(boundries_rt, x, y, amount);
    }

    private void set_boundry_gpu(int x0, int y0, int size_x, int size_y, bool is_boundry)
    {
        init_gpu();

        int new_boundry = 0;
        for (int x = x0; x < x0 + size_x; x++)
        {
            for (int y = y0; y < y0 + size_y; y++)
            {
                float boundry_value = is_boundry ? 1.0f : 0.0f;
                _set_boundry_internal(x, y, boundry_value);
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
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetInt("b", b);
        solver.SetTexture(kernel_index, "x", x);
        solver.SetTexture(kernel_index, "boundries", boundries_rt);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);
    }

    private void swap_textures_gpu2d(RenderTexture x, RenderTexture x0)
    {
        init_gpu();
        
        int kernel_index = solver.FindKernel("copy_textures");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetTexture(kernel_index, "copy_source_texture", x);
        solver.SetTexture(kernel_index, "copy_target_texture", x0);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);
    }

    /// <summary>
    /// simulate diffusion, GPU mode
    /// </summary>
    private void diffuse_gpu2d(int b, RenderTexture x, RenderTexture x0)
    {
        init_gpu();

        float a = dt * diff * (size.x - 2) * (size.y - 2);
        lin_solve_gpu2d(b, x, x0, a, 1 + 4 * a);
    }

    private RenderTexture lin_solve_gpu2d_temp;
    private ComputeBuffer lin_solve_gpu2d_maxdelta;
    private ComputeBuffer lin_solve_gpu2d_skipped_counter;

    /// <summary>
    /// solve linear system, GPU mode
    /// </summary>
    private void lin_solve_gpu2d(int b, RenderTexture x, RenderTexture x0, float a, float c)
    {
        init_gpu();
        
        bool should_reinitialize = lin_solve_gpu2d_temp == null;
        if (!should_reinitialize) should_reinitialize = should_reinitialize || lin_solve_gpu2d_temp.width != size.x || lin_solve_gpu2d_temp.height != size.y;

        if (should_reinitialize)
        {
            if (lin_solve_gpu2d_temp != null) lin_solve_gpu2d_temp.Release();
            lin_solve_gpu2d_temp = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            lin_solve_gpu2d_temp.enableRandomWrite = true;
            lin_solve_gpu2d_temp.filterMode = FilterMode.Bilinear;
        }

        if (lin_solve_gpu2d_maxdelta == null)
            lin_solve_gpu2d_maxdelta = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
        //lin_solve_gpu2d_maxdelta.SetData(new int[] { -1 });

        if (lin_solve_gpu2d_skipped_counter == null)
            lin_solve_gpu2d_skipped_counter = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
        //lin_solve_gpu2d_skipped_counter.SetData(new int[] { 0 });

        for (int i = 0; i < iter; i++)
        {
            if (linear_solver_method == GPULinearSolver.JACOBI)
            {
                int kernel_index = solver.FindKernel("copy_textures");
                solver.SetTexture(kernel_index, "copy_source_texture", x);
                solver.SetTexture(kernel_index, "copy_target_texture", lin_solve_gpu2d_temp);
                solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

                kernel_index = solver.FindKernel("lin_solve_jacobi");
                solver.SetInt("size_x", size.x);
                solver.SetInt("size_y", size.y);
                solver.SetInt("size_z", size.z);
                solver.SetInt("b", b);
                solver.SetFloat("a", a);
                solver.SetFloat("c", c);
                solver.SetTexture(kernel_index, "x", x);
                solver.SetTexture(kernel_index, "x0", x0);
                solver.SetTexture(kernel_index, "x_old", lin_solve_gpu2d_temp);
                solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);
            }
            if (linear_solver_method == GPULinearSolver.GAUSS_SEISEL)
            {

                int kernel_index = solver.FindKernel("lin_solve_gauss_seidel");
                solver.SetInt("size_x", size.x);
                solver.SetInt("size_y", size.y);
                solver.SetInt("size_z", size.z);
                solver.SetInt("b", b);
                solver.SetFloat("a", a);
                solver.SetFloat("c", c);
                solver.SetTexture(kernel_index, "x", x);
                solver.SetTexture(kernel_index, "x0", x0);
                //solver.SetBuffer(kernel_index, "lin_solve_max_delta", lin_solve_gpu2d_maxdelta);
                //solver.SetBuffer(kernel_index, "lin_solve_skipped_counter", lin_solve_gpu2d_skipped_counter);
                solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

            }

            set_bnd_gpu2d(b, x);
        }

        //int[] skipped_iterations = new int[1];
        //lin_solve_gpu2d_skipped_counter.GetData(skipped_iterations, 0, 0, 1);
        //Debug.Log(skipped_iterations[0]);
    }

    /// <summary>
    /// add force to the entirety of space, GPU mode
    /// </summary>
    private void add_external_force_gpu2d(float force, RenderTexture d)
    {
        init_gpu();

        int kernel_index = solver.FindKernel("add_external_force");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetFloat("force", force);
        solver.SetFloat("dt", dt);
        solver.SetTexture(kernel_index, "d", d);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        set_bnd_gpu2d(0, d);
    }

    /// <summary>
    /// regulate velocities to achieve zero divergence everywhere, GPU mode
    /// </summary>
    private void project_gpu2d(RenderTexture velocX, RenderTexture velocY, RenderTexture p, RenderTexture div)
    {
        init_gpu();

        
        //init_gpu();
        //
        //int kernel_index = solver.FindKernel("project_1");
        //solver.SetInt("size_x", size.x);
        //solver.SetInt("size_y", size.y);
        //solver.SetInt("size_z", size.z);
        //solver.SetTexture(kernel_index, "p", p);
        //solver.SetTexture(kernel_index, "div", div);
        //solver.SetTexture(kernel_index, "velocity_x_1", velocX);
        //solver.SetTexture(kernel_index, "velocity_y_1", velocY);
        //solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);
        //
        //set_bnd_gpu2d(0, div);
        //set_bnd_gpu2d(0, p);
        //lin_solve_gpu2d(0, p, div, 1, 4);
        //
        //kernel_index = solver.FindKernel("project_2");
        //solver.SetInt("size_x", size.x);
        //solver.SetInt("size_y", size.y);
        //solver.SetInt("size_z", size.z);
        //solver.SetTexture(kernel_index, "p", p);
        //solver.SetTexture(kernel_index, "div", div);
        //solver.SetTexture(kernel_index, "velocity_x_1", velocX);
        //solver.SetTexture(kernel_index, "velocity_y_1", velocY);
        //solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);
        //
        //set_bnd_gpu2d(1, velocX);
        //set_bnd_gpu2d(2, velocY);
    }

    private RenderTexture vx_copy_advect_gpu2d;
    private RenderTexture vy_copy_advect_gpu2d;

    /// <summary>
    /// flow given field in the direction of velocity fields, GPU mode
    /// </summary>
    private void advect_gpu2d(int b, RenderTexture d, RenderTexture d0, RenderTexture velocX, RenderTexture velocY)
    {
        init_gpu();

        bool should_reinitialize_temps = vx_copy_advect_gpu2d == null || vy_copy_advect_gpu2d == null;
        if (!should_reinitialize_temps)
        {
            bool dimention_mismatch =   vx_copy_advect_gpu2d.width != size.x || vx_copy_advect_gpu2d.height != size.y ||
                                        vy_copy_advect_gpu2d.width != size.x || vy_copy_advect_gpu2d.height != size.y;

            should_reinitialize_temps = should_reinitialize_temps || dimention_mismatch;
        }

        if (should_reinitialize_temps)
        {
            if (vx_copy_advect_gpu2d != null) vx_copy_advect_gpu2d.Release();
            vx_copy_advect_gpu2d = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            vx_copy_advect_gpu2d.enableRandomWrite = true;
            vx_copy_advect_gpu2d.filterMode = FilterMode.Bilinear;

            if (vy_copy_advect_gpu2d != null) vy_copy_advect_gpu2d.Release();
            vy_copy_advect_gpu2d = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            vy_copy_advect_gpu2d.enableRandomWrite = true;
            vy_copy_advect_gpu2d.filterMode = FilterMode.Bilinear;
        }

        int kernel_index = solver.FindKernel("copy_textures");
        solver.SetTexture(kernel_index, "copy_source_texture", velocX);
        solver.SetTexture(kernel_index, "copy_target_texture", vx_copy_advect_gpu2d);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        kernel_index = solver.FindKernel("copy_textures");
        solver.SetTexture(kernel_index, "copy_source_texture", velocY);
        solver.SetTexture(kernel_index, "copy_target_texture", vy_copy_advect_gpu2d);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        kernel_index = solver.FindKernel("advect");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetInt("b", b);
        solver.SetFloat("dt", dt);
        solver.SetTexture(kernel_index, "d", d);
        solver.SetTexture(kernel_index, "d0", d0);
        solver.SetTexture(kernel_index, "velocity_x_1", vx_copy_advect_gpu2d);
        solver.SetTexture(kernel_index, "velocity_y_1", vy_copy_advect_gpu2d);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

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

        _write_to_texture_3d(density_rt, x, y, z, amount, true);
    }

    /// <summary>
    /// add velocity to GPU buffer
    /// </summary>
    private void add_velocity_gpu3d(int x, int y, int z, float amount_x, float amount_y, float amount_z)
    {
        init_gpu();

        _write_to_texture_3d(Vx_rt, x, y, z, amount_x, true);
        _write_to_texture_3d(Vy_rt, x, y, z, amount_y, true);
        _write_to_texture_3d(Vz_rt, x, y, z, amount_z, true);
    }

    private void _set_boundry_internal3d(int x, int y, int z, float amount)
    {
        _write_to_texture_3d(boundries_rt, x, y, z, amount);
    }

    private void set_boundry_gpu3d(int x0, int y0, int z0, int size_x, int size_y, int size_z, bool is_boundry)
    {
        init_gpu();

        int new_boundry = 0;
        for (int x = x0; x < x0 + size_x; x++)
        {
            for (int y = y0; y < y0 + size_y; y++)
            {
                for (int z = z0; z < z0 + size_z; z++)
                {
                    float boundry_value = is_boundry ? 1.0f : 0.0f;
                    _set_boundry_internal3d(x, y, z, boundry_value);
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
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetInt("b", b);
        solver3D.SetTexture(kernel_index, "x", x);
        solver3D.SetTexture(kernel_index, "boundries", boundries_rt);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.y / solver3D_kernel_size.z));

        set_bnd_total_cost_3d += Time.realtimeSinceStartup * 1000 - set_bnd_begin_time;
        set_bnd_call_count_3d++;
    }

    /// <summary>
    /// simulate diffusion, GPU mode
    /// </summary>
    private void diffuse_gpu3d(int b, RenderTexture x, RenderTexture x0)
    {
        init_gpu();

        float a = dt * diff * (size.x - 2) * (size.y - 2);
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
            if (temp_linear_solve_jacobi == null || temp_linear_solve_jacobi.width != size.x || temp_linear_solve_jacobi.height != size.y || temp_linear_solve_jacobi.volumeDepth != size.z)
            {
                if (temp_linear_solve_jacobi != null) temp_linear_solve_jacobi.Release();
                temp_linear_solve_jacobi = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
                temp_linear_solve_jacobi.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                temp_linear_solve_jacobi.volumeDepth = size.z;
                temp_linear_solve_jacobi.enableRandomWrite = true;
                temp_linear_solve_jacobi.filterMode = FilterMode.Bilinear;
            }

            int copy_kernel_index = solver3D.FindKernel("copy_textures_3d");
            solver3D.SetTexture(copy_kernel_index, "copy_source_texture", x);
            solver3D.SetTexture(copy_kernel_index, "copy_target_texture", temp_linear_solve_jacobi);

            int lin_solve_kernel_index = solver3D.FindKernel("lin_solve_jacobi_3d");
            solver3D.SetInt("size_x", size.x);
            solver3D.SetInt("size_y", size.y);
            solver3D.SetInt("size_z", size.z);
            solver3D.SetInt("b", b);
            solver3D.SetFloat("a", a);
            solver3D.SetFloat("c", c);
            solver3D.SetTexture(lin_solve_kernel_index, "x", x);
            solver3D.SetTexture(lin_solve_kernel_index, "x0", x0);
            solver3D.SetTexture(lin_solve_kernel_index, "x_old", temp_linear_solve_jacobi);

            for (int i = 0; i < iter; i++)
            {
                float copy_textures_begin_time = Time.realtimeSinceStartup * 1000;

                solver3D.Dispatch(copy_kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

                copy_textures_total_cost_3d += Time.realtimeSinceStartup * 1000 - copy_textures_begin_time;
                copy_textures_call_count_3d++;

                solver3D.Dispatch(lin_solve_kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));
            }
        }

        if (linear_solver_method == GPULinearSolver.GAUSS_SEISEL)
        {
            int kernel_index = solver3D.FindKernel("lin_solve_gauss_seidel_3d");
            solver3D.SetInt("size_x", size.x);
            solver3D.SetInt("size_y", size.y);
            solver3D.SetInt("size_z", size.z);
            solver3D.SetInt("b", b);
            solver3D.SetFloat("a", a);
            solver3D.SetFloat("c", c);
            solver3D.SetTexture(kernel_index, "x", x);
            solver3D.SetTexture(kernel_index, "x0", x0);
            for (int i = 0; i < iter; i++)
                solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));
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
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetFloat("force", force);
        solver3D.SetFloat("dt", dt);
        solver3D.SetTexture(kernel_index, "d", d);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

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
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetTexture(kernel_index, "p", p);
        solver3D.SetTexture(kernel_index, "div", div);
        solver3D.SetTexture(kernel_index, "velocity_x_1", velocX);
        solver3D.SetTexture(kernel_index, "velocity_y_1", velocY);
        solver3D.SetTexture(kernel_index, "velocity_z_1", velocZ);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        set_bnd_gpu3d(0, div);
        set_bnd_gpu3d(0, p);
        lin_solve_gpu3d(0, p, div, 1, 6);

        kernel_index = solver3D.FindKernel("project_2_3d");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetTexture(kernel_index, "p", p);
        solver3D.SetTexture(kernel_index, "div", div);
        solver3D.SetTexture(kernel_index, "velocity_x_1", velocX);
        solver3D.SetTexture(kernel_index, "velocity_y_1", velocY);
        solver3D.SetTexture(kernel_index, "velocity_z_1", velocZ);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

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
        if (!should_initialize_temps) 
            should_initialize_temps = should_initialize_temps || 
                            temp_advect_vx.width        != size.x   || temp_advect_vy.width         != size.x   || temp_advect_vz.width         != size.x   ||
                            temp_advect_vx.height       != size.y   || temp_advect_vy.height        != size.y   || temp_advect_vz.height        != size.y   ||
                            temp_advect_vx.volumeDepth  != size.z   || temp_advect_vy.volumeDepth   != size.z   || temp_advect_vz.volumeDepth   != size.z;

        int kernel_index;

        if (should_initialize_temps)
        {
            if (temp_advect_vx != null) temp_advect_vx.Release();
            if (temp_advect_vy != null) temp_advect_vy.Release();
            if (temp_advect_vz != null) temp_advect_vz.Release();

            temp_advect_vx = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            temp_advect_vx.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            temp_advect_vx.volumeDepth = size.z;
            temp_advect_vx.enableRandomWrite = true;
            temp_advect_vx.filterMode = FilterMode.Bilinear;

            temp_advect_vy = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            temp_advect_vy.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            temp_advect_vy.volumeDepth = size.z;
            temp_advect_vy.enableRandomWrite = true;
            temp_advect_vy.filterMode = FilterMode.Bilinear;

            temp_advect_vz = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            temp_advect_vz.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            temp_advect_vz.volumeDepth = size.z;
            temp_advect_vz.enableRandomWrite = true;
            temp_advect_vz.filterMode = FilterMode.Bilinear;
        }

        float texture_copies_begin_time = Time.realtimeSinceStartup * 1000;

        kernel_index = solver3D.FindKernel("copy_textures_3d");
        solver3D.SetTexture(kernel_index, "copy_source_texture", velocX);
        solver3D.SetTexture(kernel_index, "copy_target_texture", temp_advect_vx);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        kernel_index = solver3D.FindKernel("copy_textures_3d");
        solver3D.SetTexture(kernel_index, "copy_source_texture", velocY);
        solver3D.SetTexture(kernel_index, "copy_target_texture", temp_advect_vy);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        kernel_index = solver3D.FindKernel("copy_textures_3d");
        solver3D.SetTexture(kernel_index, "copy_source_texture", velocZ);
        solver3D.SetTexture(kernel_index, "copy_target_texture", temp_advect_vz);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        copy_textures_total_cost_3d += Time.realtimeSinceStartup * 1000 - texture_copies_begin_time;
        copy_textures_call_count_3d += 3;

        kernel_index = solver3D.FindKernel("advect_3d");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetInt("b", b);
        solver3D.SetFloat("dt", dt);
        solver3D.SetTexture(kernel_index, "d", d);
        solver3D.SetTexture(kernel_index, "d0", d0);
        solver3D.SetTexture(kernel_index, "velocity_x_1", temp_advect_vx);
        solver3D.SetTexture(kernel_index, "velocity_y_1", temp_advect_vy);
        solver3D.SetTexture(kernel_index, "velocity_z_1", temp_advect_vz);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        set_bnd_gpu3d(b, d);

        advect_total_cost_3d += Time.realtimeSinceStartup * 1000 - advect_begin_time;
        advect_call_count_3d++;
    }

    #endregion

    /// <summary>
    /// set given rectangular area as boundry to current mode's buffer
    /// </summary>
    public void set_boundry(int x0, int y0, int size_x, int size_y, bool is_boundry)
    {
        if (mode == ComputationMode.GPU3D) return;
        if (mode == ComputationMode.GPU2D) set_boundry_gpu(x0, y0, size_x, size_y, is_boundry);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_density(int x, int y, float amount)
    {
        if (mode == ComputationMode.GPU3D) return;
        if (mode == ComputationMode.GPU2D) add_density_gpu(x, y, amount);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_velocity(int x, int y, float amount_x, float amount_y)
    {
        if (mode == ComputationMode.GPU3D) return;
        if (mode == ComputationMode.GPU2D) add_velocity_gpu(x, y, amount_x, amount_y);
    }

    /// <summary>
    /// set given rectangular area as boundry to current mode's buffer
    /// </summary>
    public void set_boundry(int x0, int y0, int z0, int size_x, int size_y, int size_z, bool is_boundry)
    {
        if (mode == ComputationMode.GPU2D) set_boundry_gpu(x0, y0, size_x, size_y, is_boundry);
        if (mode == ComputationMode.GPU3D) set_boundry_gpu3d(x0, y0, z0, size_x, size_y, size_z, is_boundry);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_density(int x, int y, int z, float amount)
    {
        if (mode == ComputationMode.GPU2D) add_density_gpu(x, y, amount);
        if (mode == ComputationMode.GPU3D) add_density_gpu3d(x, y, z, amount);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_velocity(int x, int y, int z, float amount_x, float amount_y, float amount_z)
    {
        if (mode == ComputationMode.GPU2D) add_velocity_gpu(x, y, amount_x, amount_y);
        if (mode == ComputationMode.GPU3D) add_velocity_gpu3d(x, y, z, amount_x, amount_y, amount_z);
    }

    public void add_density_worldcoord(Vector3 world_coord, float amount)
    {
        Vector3 pixel_coord = (world_coord - transform.position + transform.lossyScale / 2);
        pixel_coord.x *= size.x / transform.lossyScale.x;
        pixel_coord.y *= size.y / transform.lossyScale.y;
        pixel_coord.z *= size.z / transform.lossyScale.z;
        add_density((int)pixel_coord.x, (int)pixel_coord.y, (int)pixel_coord.z, amount);
    }

    public void add_velocity_worldcoord(Vector3 world_coord, Vector3 velocity)
    {
        Vector3 pixel_coord = (world_coord - transform.position + transform.lossyScale / 2);
        pixel_coord.x *= size.x / transform.lossyScale.x;
        pixel_coord.y *= size.y / transform.lossyScale.y;
        pixel_coord.z *= size.z / transform.lossyScale.z;
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
            if (mode == ComputationMode.GPU3D) step_gpu_3d();
            else if (mode == ComputationMode.GPU2D) step_gpu();
        }
    }

    public Vector3Int simulation_resolution;
    public int simulation_iteration_per_step;
    public float diffusion;

    public int wall_thickness;
    void Start()
    {
        init_parameters(simulation_resolution, simulation_iteration_per_step, 0.1f, 0.0001f * diffusion, 0.1f);

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

    public bool should_update_boundries;

    void update_solid_boundries()
    {
        if (Time.realtimeSinceStartup < 5) return;

        if (!should_update_boundries) return;
        should_update_boundries = false;

        for (int z = 0; z < size.z; z++){
            for (int y = 0; y < size.y; y++){
                for (int x = 0; x < size.x; x++){

                    Vector3 pixel_coord = new Vector3(x, y, z);
                    Vector3 world_coord = new Vector3(
                        (float)x / size.x * transform.lossyScale.x - transform.lossyScale.x / 2 + transform.position.x,
                        (float)y / size.y * transform.lossyScale.y - transform.lossyScale.y / 2 + transform.position.y,
                        (float)z / size.z * transform.lossyScale.z - transform.lossyScale.z / 2 + transform.position.z
                        );
                    Vector3 voxel_size = new Vector3(transform.lossyScale.x / size.x, transform.lossyScale.y / size.y, transform.lossyScale.z / size.z);

                    Collider[] hits = Physics.OverlapBox(world_coord, voxel_size / 2);
                    bool is_boundry = false;

                    foreach (Collider hit in hits)
                    {
                        is_boundry = hit.GetComponent<SmokeProof>() != null;
                        if (is_boundry) break;
                    }
                    
                    set_boundry(x, y, z, 2, 2, 2, is_boundry);
                }
            }
        }

        set_boundry(0, 0, 0, wall_thickness, size.y, size.z, true);
        set_boundry(0, 0, 0, size.x, wall_thickness, size.z, true);
        if (mode == ComputationMode.GPU3D) set_boundry(0, 0, 0, size.x, size.y, wall_thickness, true);

        set_boundry(size.x - wall_thickness, 0, 0, wall_thickness, size.y, size.z, true);
        set_boundry(0, size.y - wall_thickness, 0, size.x, wall_thickness, size.z, true);
        if (mode == ComputationMode.GPU3D) set_boundry(0, 0, size.z - wall_thickness, size.x, size.y, wall_thickness, true);

    }

    //private void OnDrawGizmos()
    //{
    //    for (int z = 0; z < size.z; z++)
    //    {
    //        for (int y = 0; y < size.y; y++)
    //        {
    //            for (int x = 0; x < size.x; x++)
    //            {
    //
    //                Vector3 pixel_coord = new Vector3(x, y, z);
    //                Vector3 world_coord = new Vector3(
    //                    (float)x / size.x * transform.lossyScale.x - transform.lossyScale.x / 2,
    //                    (float)y / size.y * transform.lossyScale.y - transform.lossyScale.y / 2,
    //                    (float)z / size.z * transform.lossyScale.z - transform.lossyScale.z / 2
    //                    );
    //                Vector3 voxel_size = new Vector3(transform.lossyScale.x / size.x, transform.lossyScale.y / size.y, transform.lossyScale.z / size.z);
    //
    //                Gizmos.DrawWireCube(world_coord, voxel_size);
    //            }
    //        }
    //    }
    //}

    void FixedUpdate()
    {
        init_parameters(simulation_resolution, simulation_iteration_per_step, 0.1f, 0.0001f * diffusion, 0.1f);

        update_solid_boundries();

        if (Input.GetKey(KeyCode.Mouse0))
        {
            int cx = (int)(Input.mousePosition.x / Screen.width * size.x);
            int cy = (int)(Input.mousePosition.y / Screen.height * size.y);
            int z = 0;
            if (mode == ComputationMode.GPU3D) z = size.z / 4 * 3;
            for (int i = 0; i < mouse_add_size_density; i++)
                for (int j = 0; j < mouse_add_size_density; j++)
                    add_density(cx + i, cy + j, z, mouse_add_density);

            for (int i = 0; i < mouse_add_size_velocity; i++)
                for (int j = 0; j < mouse_add_size_velocity; j++)
                    add_velocity(cx + i, cy + j, z, mouse_add_velocity.x, mouse_add_velocity.y, 0);
        }

        if (Input.GetKey(KeyCode.Space))
        {
            int z = 0;
            if (mode == ComputationMode.GPU3D) z = size.z / 2;
            for (int i = 0; i < mouse_add_size_density; i++)
                for (int j = 0; j < mouse_add_size_density; j++)
                    for (int k = 0; k < mouse_add_size_density; k++)
                    {
                        int offset = size.x / 3;
                        add_density(size.x / 2 + i - offset, size.y / 4 + j, z + k, mouse_add_density);
                        add_density(size.x / 2 + i + offset, size.y / 4 + j, z + k, mouse_add_density);
                    }

            for (int i = 0; i < mouse_add_size_velocity; i++)
                for (int j = 0; j < mouse_add_size_velocity; j++)
                    for (int k = 0; k < mouse_add_size_velocity; k++)
                    {
                        int offset = size.x / 3;
                        add_velocity(size.x / 2 + i + offset, size.y / 4 + j, z + k, -4, 0, Random.value - 0.5f);
                        add_velocity(size.x / 2 + i - offset, size.y / 4 + j, z + k, +4, 0, Random.value - 0.5f);
                    }
        }


    }


    public bool iterate_time;
    private void Update()
    {
        if (iterate_time)
            step(Time.deltaTime * 1000);
    }

    bool _first_draw_in_gpu_mode;

    // texture to display cpu mode buffers
    RenderTexture slice_of_3D;
    public int displaying_slice_index;
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (_first_draw_in_gpu_mode)
        {
            init_gpu();
            if (slice_of_3D) slice_of_3D.Release();
            slice_of_3D = new RenderTexture(size.x, size.y, 0);
            slice_of_3D.filterMode = FilterMode.Bilinear;
            slice_of_3D.enableRandomWrite = true;
        }
        _first_draw_in_gpu_mode = false;

        if (mode == ComputationMode.GPU2D)
            Graphics.Blit(density_rt, (RenderTexture)null);

        if (mode == ComputationMode.GPU3D && _gpu_buffers_sized_for_3D)
        {
            int kernel_index = solver3D.FindKernel("get_3d_texture_slice");
            solver3D.SetInt("slice_z_index", displaying_slice_index);
            solver3D.SetTexture(kernel_index, "copy_source_texture", density_rt);
            solver3D.SetTexture(kernel_index, "slice_target", slice_of_3D);
            solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), 1);
            Graphics.Blit(slice_of_3D, (RenderTexture)null);
        }
    }
}
