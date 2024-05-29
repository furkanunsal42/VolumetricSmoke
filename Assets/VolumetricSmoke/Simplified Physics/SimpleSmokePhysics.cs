using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleSmokePhysics : MonoBehaviour
{
    public static HashSet<SimpleSmokePhysics> all_instances = new HashSet<SimpleSmokePhysics>();

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

    public Vector3 gravity = new Vector3(0, -9.8f, 0);

    private float dt;

    // GPU mode buffers
    /*[HideInInspector]*/
    public RenderTexture density1_rt;
    public RenderTexture density2_rt;
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

    public SimpleSmokePhysics(Vector3Int size, float dt)
    {
        init_parameters(size, dt);
        init_gpu();
    }

    private void free_vram()
    {
        if (density1_rt != null) density1_rt.Release();
        if (density2_rt != null) density2_rt.Release();
        if (slice_of_3D != null) slice_of_3D.Release();
    }

    public void OnDestroy()
    {
        free_vram();
    }

    /// <summary>
    /// initialize simulation parameters
    /// </summary>
    public void init_parameters(Vector3Int size, float dt)
    {
        solver3D_kernel_size = new Vector3(2, 2, 2);

        if (size != this.size)
        {
            _gpu_buffers_initialized = false;
            _first_draw_in_gpu_mode = true;
            _gpu_buffers_sized_for_3D = false;
        }

        this.size = size;
        this.dt = dt;
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

            this.density1_rt =  new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.density2_rt =  new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.boundries_rt = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);

            this.density1_rt.dimension  = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.density2_rt.dimension  = UnityEngine.Rendering.TextureDimension.Tex3D;
            this.boundries_rt.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;

            this.density1_rt.volumeDepth    = size.z;
            this.density2_rt.volumeDepth    = size.z;
            this.boundries_rt.volumeDepth   = size.z;

            this.density1_rt.wrapMode   = TextureWrapMode.Clamp;
            this.density2_rt.wrapMode   = TextureWrapMode.Clamp;
            this.boundries_rt.wrapMode  = TextureWrapMode.Clamp;

            this.density1_rt.enableRandomWrite   = true;
            this.density2_rt.enableRandomWrite   = true;
            this.boundries_rt.enableRandomWrite = true;

            this.density1_rt.filterMode     = FilterMode.Bilinear;
            this.density2_rt.filterMode     = FilterMode.Bilinear;
            this.boundries_rt.filterMode    = FilterMode.Bilinear;
        }
        else
        {
            _gpu_buffers_sized_for_3D = false;

            Debug.Log("2d gpu init");

            if(this.density1_rt  ) this.density1_rt.Release();
            if(this.density2_rt  ) this.density2_rt.Release();
            if(this.boundries_rt) this.boundries_rt.Release();

            this.density1_rt    = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.density2_rt    = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            this.boundries_rt   = new RenderTexture(size.x, size.y, 0, RenderTextureFormat.RFloat, 0);
            
            this.density1_rt.wrapMode   = TextureWrapMode.Clamp;
            this.density2_rt.wrapMode   = TextureWrapMode.Clamp;
            this.boundries_rt.wrapMode  = TextureWrapMode.Clamp;

            this.density1_rt.enableRandomWrite  = true;
            this.density2_rt.enableRandomWrite  = true;
            this.boundries_rt.enableRandomWrite = true;

            this.density1_rt.filterMode     = FilterMode.Bilinear;
            this.density2_rt.filterMode     = FilterMode.Bilinear;
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
    }

    #endregion

    #region GPU2D
    /*
    /// <summary>
    /// single iteration of simulation in GPU mode
    /// </summary>
    private void step_gpu()
    {
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

        int kernel_index = solver.FindKernel("project_1");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetTexture(kernel_index, "p", p);
        solver.SetTexture(kernel_index, "div", div);
        solver.SetTexture(kernel_index, "velocity_x_1", velocX);
        solver.SetTexture(kernel_index, "velocity_y_1", velocY);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        set_bnd_gpu2d(0, div);
        set_bnd_gpu2d(0, p);
        lin_solve_gpu2d(0, p, div, 1, 4);

        kernel_index = solver.FindKernel("project_2");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetTexture(kernel_index, "p", p);
        solver.SetTexture(kernel_index, "div", div);
        solver.SetTexture(kernel_index, "velocity_x_1", velocX);
        solver.SetTexture(kernel_index, "velocity_y_1", velocY);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        set_bnd_gpu2d(1, velocX);
        set_bnd_gpu2d(2, velocY);
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
    */
    #endregion

    #region GPU3D

    // privates

    /// <summary>
    /// single iteration of simulation in GPU3D mode
    /// </summary>
    private void step_gpu_3d()
    {
        float step_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();

        int kernel_index = solver3D.FindKernel("physics_step");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetFloat("dt", 0.03f);
        solver3D.SetTexture(kernel_index, "density_1", density1_rt);
        solver3D.SetTexture(kernel_index, "density_2", density2_rt);
        solver3D.SetTexture(kernel_index, "boundries", boundries_rt);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        kernel_index = solver3D.FindKernel("swap_textures_3d");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetTexture(kernel_index, "density_1", density1_rt);
        solver3D.SetTexture(kernel_index, "density_2", density2_rt);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));
    }

    /// <summary>
    /// add density to GPU buffer
    /// </summary>
    private void add_density_gpu3d(int x, int y, int z, float amount)
    {
        init_gpu();

        _write_to_texture_3d(density1_rt, x, y, z, amount, true);
        _write_to_texture_3d(density2_rt, x, y, z, amount, true);
    }

    /// <summary>
    /// add velocity to GPU buffer
    /// </summary>
    //private void add_velocity_gpu3d(int x, int y, int z, float amount_x, float amount_y, float amount_z)
    //{
    //    init_gpu();
    //
    //    _write_to_texture_3d(Vx_rt, x, y, z, amount_x, true);
    //    _write_to_texture_3d(Vy_rt, x, y, z, amount_y, true);
    //    _write_to_texture_3d(Vz_rt, x, y, z, amount_z, true);
    //}

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
    }

    #endregion

    /// <summary>
    /// set given rectangular area as boundry to current mode's buffer
    /// </summary>
    public void set_boundry(int x0, int y0, int size_x, int size_y, bool is_boundry)
    {
        if (mode == ComputationMode.GPU3D) return;
        //if (mode == ComputationMode.GPU2D) set_boundry_gpu(x0, y0, size_x, size_y, is_boundry);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_density(int x, int y, float amount)
    {
        if (mode == ComputationMode.GPU3D) return;
        //if (mode == ComputationMode.GPU2D) add_density_gpu(x, y, amount);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    //public void add_velocity(int x, int y, float amount_x, float amount_y)
    //{
    //    if (mode == ComputationMode.GPU3D) return;
    //    //if (mode == ComputationMode.GPU2D) add_velocity_gpu(x, y, amount_x, amount_y);
    //}

    /// <summary>
    /// set given rectangular area as boundry to current mode's buffer
    /// </summary>
    public void set_boundry(int x0, int y0, int z0, int size_x, int size_y, int size_z, bool is_boundry)
    {
        //if (mode == ComputationMode.GPU2D) set_boundry_gpu(x0, y0, size_x, size_y, is_boundry);
        if (mode == ComputationMode.GPU3D) set_boundry_gpu3d(x0, y0, z0, size_x, size_y, size_z, is_boundry);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_density(int x, int y, int z, float amount)
    {
        //if (mode == ComputationMode.GPU2D) add_density_gpu(x, y, amount);
        if (mode == ComputationMode.GPU3D) add_density_gpu3d(x, y, z, amount);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    //public void add_velocity(int x, int y, int z, float amount_x, float amount_y, float amount_z)
    //{
    //    if (mode == ComputationMode.GPU2D) add_velocity_gpu(x, y, amount_x, amount_y);
    //    if (mode == ComputationMode.GPU3D) add_velocity_gpu3d(x, y, z, amount_x, amount_y, amount_z);
    //}

    public void add_density_worldcoord(Vector3 world_coord, float amount)
    {
        Vector3 pixel_coord = (world_coord - transform.position + transform.lossyScale / 2);
        pixel_coord.x *= size.x / transform.lossyScale.x;
        pixel_coord.y *= size.y / transform.lossyScale.y;
        pixel_coord.z *= size.z / transform.lossyScale.z;
        add_density((int)pixel_coord.x, (int)pixel_coord.y, (int)pixel_coord.z, amount);
    }

    //public void add_velocity_worldcoord(Vector3 world_coord, Vector3 velocity)
    //{
    //    Vector3 pixel_coord = (world_coord - transform.position + transform.lossyScale / 2);
    //    pixel_coord.x *= size.x / transform.lossyScale.x;
    //    pixel_coord.y *= size.y / transform.lossyScale.y;
    //    pixel_coord.z *= size.z / transform.lossyScale.z;
    //    add_velocity((int)pixel_coord.x, (int)pixel_coord.y, (int)pixel_coord.z, velocity.x, velocity.y, velocity.z);
    //}

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
            //else if (mode == ComputationMode.GPU2D) step_gpu();
        }
    }

    public Vector3Int simulation_resolution;
    public int wall_thickness;
    void Start()
    {
        init_parameters(simulation_resolution, 0.1f);

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
        init_parameters(simulation_resolution, 0.1f);

        update_solid_boundries();

        add_density(size.x / 2, size.y / 2, size.z / 2, 100);

        if (Input.GetKey(KeyCode.Mouse0))
        {
            int cx = (int)(Input.mousePosition.x / Screen.width * size.x);
            int cy = (int)(Input.mousePosition.y / Screen.height * size.y);
            int z = 0;
            if (mode == ComputationMode.GPU3D) z = size.z / 4 * 3;
            for (int i = 0; i < mouse_add_size_density; i++)
                for (int j = 0; j < mouse_add_size_density; j++)
                    add_density(cx + i, cy + j, z, mouse_add_density);

            //for (int i = 0; i < mouse_add_size_velocity; i++)
            //    for (int j = 0; j < mouse_add_size_velocity; j++)
            //        add_velocity(cx + i, cy + j, z, mouse_add_velocity.x, mouse_add_velocity.y, 0);
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

            //for (int i = 0; i < mouse_add_size_velocity; i++)
            //    for (int j = 0; j < mouse_add_size_velocity; j++)
            //        for (int k = 0; k < mouse_add_size_velocity; k++)
            //        {
            //            int offset = size.x / 3;
            //            add_velocity(size.x / 2 + i + offset, size.y / 4 + j, z + k, -4, 0, Random.value - 0.5f);
            //            add_velocity(size.x / 2 + i - offset, size.y / 4 + j, z + k, +4, 0, Random.value - 0.5f);
            //        }
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
            Graphics.Blit(density1_rt, (RenderTexture)null);

        if (mode == ComputationMode.GPU3D && _gpu_buffers_sized_for_3D)
        {
            int kernel_index = solver3D.FindKernel("get_3d_texture_slice");
            solver3D.SetInt("slice_z_index", displaying_slice_index);
            solver3D.SetTexture(kernel_index, "copy_source_texture", density1_rt);
            solver3D.SetTexture(kernel_index, "slice_target", slice_of_3D);
            solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), 1);
            Graphics.Blit(slice_of_3D, (RenderTexture)null);
        }
    }
}
