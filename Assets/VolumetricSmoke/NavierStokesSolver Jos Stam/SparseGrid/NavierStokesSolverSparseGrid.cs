using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NavierStokesSolverSparseGrid : MonoBehaviour    // referance paper : https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
{
    public static HashSet<NavierStokesSolverSparseGrid> all_instances = new HashSet<NavierStokesSolverSparseGrid>();

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
    public ComputeBuffer s_cb;
    /*[HideInInspector]*/
    public ComputeBuffer density_cb;

    /*[HideInInspector]*/
    public ComputeBuffer Vx_cb;
    /*[HideInInspector]*/
    public ComputeBuffer Vy_cb;
    /*[HideInInspector]*/
    public ComputeBuffer Vz_cb;

    /*[HideInInspector]*/
    public ComputeBuffer Vx0_cb;
    /*[HideInInspector]*/
    public ComputeBuffer Vy0_cb;
    /*[HideInInspector]*/
    public ComputeBuffer Vz0_cb;

    /*[HideInInspector]*/
    public ComputeBuffer boundries_cb;

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

    public NavierStokesSolverSparseGrid(Vector3Int size, float dt, int iterations, float diffusion, float viscosity)
    {
        init_parameters(size, iterations, dt, diffusion, viscosity);
        init_gpu();
    }

    private void free_vram()
    {
        if (s_cb != null) s_cb.Release();
        if (density_cb != null) density_cb.Release();
        if (Vx_cb != null) Vx_cb.Release();
        if (Vy_cb != null) Vy_cb.Release();
        if (Vz_cb != null) Vz_cb.Release();
        if (Vx0_cb != null) Vx0_cb.Release();
        if (Vy0_cb != null) Vy0_cb.Release();
        if (Vz0_cb != null) Vz0_cb.Release();
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

            this.s_cb =         new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            this.density_cb =   new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            this.Vx_cb =        new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            this.Vy_cb =        new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            this.Vz_cb =        new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            this.Vx0_cb =       new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            this.Vy0_cb =       new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            this.Vz0_cb =       new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            this.boundries_cb = new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);

        }
        else{

            _gpu_buffers_sized_for_3D = false;

            Debug.Log("2d gpu init");

            free_vram();

            this.s_cb           = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
            this.density_cb     = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
            this.Vx_cb          = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
            this.Vy_cb          = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
            this.Vz_cb          = null;
            this.Vx0_cb         = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
            this.Vy0_cb         = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
            this.Vz0_cb         = null;
            this.boundries_cb   = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
        }                  
    }

    #region UTILS
    private int _interleave_with_zero(int input)
    {
        input = (input ^ (input << 8)) & 0x00ff00ff;
        input = (input ^ (input << 4)) & 0x0f0f0f0f;
        input = (input ^ (input << 2)) & 0x33333333;
        input = (input ^ (input << 1)) & 0x55555555;
        return input;
    }

    private int _interleave(int x, int y)
    {
        return _interleave_with_zero(x) | (_interleave_with_zero(y) << 1);
    }

    private int _interleave(int x, int y, int z)
    {
        return _interleave_with_zero(x) | (_interleave_with_zero(y) << 1) | (_interleave_with_zero(z) << 2);
    }

    /// <summary>
    /// add the given amount to a single index of a GPU buffer
    /// </summary>
    private void _write_to_texture(ComputeBuffer target, int x, int y, float amount, bool block_writes_to_edges = false)
    {
        int upper_x_limit = block_writes_to_edges ? size.x - 2 : size.x - 1;
        int upper_y_limit = block_writes_to_edges ? size.y - 2 : size.y - 1;
        int lower_x_limit = block_writes_to_edges ? 1 : 0;
        int lower_y_limit = block_writes_to_edges ? 1 : 0;

        if (x > upper_x_limit || x < lower_x_limit) return;
        if (y > upper_y_limit || y < lower_y_limit) return;

        target.SetData(new float[] { amount }, 0, _interleave(x, y), 1);
    }

    /// <summary>
    /// add the given amount to a single index of a GPU3D buffer
    /// </summary>
    private void _write_to_texture_3d(ComputeBuffer target, int x, int y, int z, float amount, bool block_writes_to_edges = false)
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

        target.SetData(new float[] { amount }, 0, _interleave(x, y, z), 1);

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

        add_external_force_gpu2d(gravity.x * 0.0001f, Vx_cb);
        add_external_force_gpu2d(gravity.y * 0.0001f, Vy_cb);

        diffuse_gpu2d(1, Vx0_cb, Vx_cb);
        diffuse_gpu2d(2, Vy0_cb, Vy_cb);

        project_gpu2d(Vx0_cb, Vy0_cb, Vx_cb, Vy_cb);

        advect_gpu2d(1, Vx_cb, Vx0_cb, Vx0_cb, Vy0_cb);
        advect_gpu2d(2, Vy_cb, Vy0_cb, Vx0_cb, Vy0_cb);

        project_gpu2d(Vx_cb, Vy_cb, Vx0_cb, Vy0_cb);

        diffuse_gpu2d(0, s_cb, density_cb);
        advect_gpu2d(0, density_cb, s_cb, Vx_cb, Vy_cb);
    }

    /// <summary>
    /// add density to GPU buffer
    /// </summary>
    private void add_density_gpu(int x, int y, float amount)
    {
        init_gpu();
        _write_to_texture(density_cb, x, y, amount, true);
    }

    /// <summary>
    /// add velocity to GPU buffer
    /// </summary>
    private void add_velocity_gpu(int x, int y, float amount_x, float amount_y)
    {
        init_gpu();

        _write_to_texture(Vx_cb, x, y, amount_x, true);
        _write_to_texture(Vy_cb, x, y, amount_y, true);
    }

    private void _set_boundry_internal(int x, int y, float amount)
    {
        _write_to_texture(boundries_cb, x, y, amount);
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
    private void set_bnd_gpu2d(int b, ComputeBuffer x)
    {
        init_gpu();

        int kernel_index = solver.FindKernel("set_bnd");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetInt("b", b);
        solver.SetBuffer(kernel_index, "x", x);
        solver.SetBuffer(kernel_index, "boundries", boundries_cb);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);
    }

    /// <summary>
    /// simulate diffusion, GPU mode
    /// </summary>
    private void diffuse_gpu2d(int b, ComputeBuffer x, ComputeBuffer x0)
    {
        init_gpu();

        float a = dt * diff * (size.x - 2) * (size.y - 2);
        lin_solve_gpu2d(b, x, x0, a, 1 + 4 * a);
    }

    private ComputeBuffer lin_solve_gpu2d_temp;
    private ComputeBuffer lin_solve_gpu2d_maxdelta;
    private ComputeBuffer lin_solve_gpu2d_skipped_counter;

    /// <summary>
    /// solve linear system, GPU mode
    /// </summary>
    private void lin_solve_gpu2d(int b, ComputeBuffer x, ComputeBuffer x0, float a, float c)
    {
        init_gpu();
        
        bool should_reinitialize = lin_solve_gpu2d_temp == null;
        if (!should_reinitialize) should_reinitialize = should_reinitialize || lin_solve_gpu2d_temp.count != size.x * size.y;

        if (should_reinitialize)
        {
            if (lin_solve_gpu2d_temp != null) lin_solve_gpu2d_temp.Release();
            lin_solve_gpu2d_temp = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
        }

        if (lin_solve_gpu2d_maxdelta == null)
            lin_solve_gpu2d_maxdelta = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
        //lin_solve_gpu2d_maxdelta.SetData(new int[] { -1 });
        
        if (lin_solve_gpu2d_skipped_counter == null)
            lin_solve_gpu2d_skipped_counter = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
        //lin_solve_gpu2d_skipped_counter.SetData(new int[] { 0 });

        for (int i = 0; i < iter; i++)
        {
            //if (linear_solver_method == GPULinearSolver.JACOBI)
            //{
            //    int kernel_index = solver.FindKernel("copy_textures");
            //    solver.SetTexture(kernel_index, "copy_source_texture", x);
            //    solver.SetTexture(kernel_index, "copy_target_texture", lin_solve_gpu2d_temp);
            //    solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);
            //
            //    kernel_index = solver.FindKernel("lin_solve_jacobi");
            //    solver.SetInt("size_x", size.x);
            //    solver.SetInt("size_y", size.y);
            //    solver.SetInt("size_z", size.z);
            //    solver.SetInt("b", b);
            //    solver.SetFloat("a", a);
            //    solver.SetFloat("c", c);
            //    solver.SetTexture(kernel_index, "x", x);
            //    solver.SetTexture(kernel_index, "x0", x0);
            //    solver.SetTexture(kernel_index, "x_old", lin_solve_gpu2d_temp);
            //    solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);
            //}
            if (linear_solver_method == GPULinearSolver.GAUSS_SEISEL)
            {

                int kernel_index = solver.FindKernel("lin_solve_gauss_seidel");
                solver.SetInt("size_x", size.x);
                solver.SetInt("size_y", size.y);
                solver.SetInt("size_z", size.z);
                solver.SetInt("b", b);
                solver.SetFloat("a", a);
                solver.SetFloat("c", c);
                solver.SetBuffer(kernel_index, "x", x);
                solver.SetBuffer(kernel_index, "x0", x0);
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
    private void add_external_force_gpu2d(float force, ComputeBuffer d)
    {
        init_gpu();

        int kernel_index = solver.FindKernel("add_external_force");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetFloat("force", force);
        solver.SetFloat("dt", dt);
        solver.SetBuffer(kernel_index, "d", d);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        set_bnd_gpu2d(0, d);
    }

    /// <summary>
    /// regulate velocities to achieve zero divergence everywhere, GPU mode
    /// </summary>
    private void project_gpu2d(ComputeBuffer velocX, ComputeBuffer velocY, ComputeBuffer p, ComputeBuffer div)
    {
        init_gpu();

        int kernel_index = solver.FindKernel("project_1");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetBuffer(kernel_index, "p", p);
        solver.SetBuffer(kernel_index, "div", div);
        solver.SetBuffer(kernel_index, "velocity_x_1", velocX);
        solver.SetBuffer(kernel_index, "velocity_y_1", velocY);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        set_bnd_gpu2d(0, div);
        set_bnd_gpu2d(0, p);
        lin_solve_gpu2d(0, p, div, 1, 4);

        kernel_index = solver.FindKernel("project_2");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetBuffer(kernel_index, "p", p);
        solver.SetBuffer(kernel_index, "div", div);
        solver.SetBuffer(kernel_index, "velocity_x_1", velocX);
        solver.SetBuffer(kernel_index, "velocity_y_1", velocY);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        set_bnd_gpu2d(1, velocX);
        set_bnd_gpu2d(2, velocY);
    }

    private ComputeBuffer vx_copy_advect_gpu2d;
    private ComputeBuffer vy_copy_advect_gpu2d;

    /// <summary>
    /// flow given field in the direction of velocity fields, GPU mode
    /// </summary>
    private void advect_gpu2d(int b, ComputeBuffer d, ComputeBuffer d0, ComputeBuffer velocX, ComputeBuffer velocY)
    {
        init_gpu();

        bool should_reinitialize_temps = vx_copy_advect_gpu2d == null || vy_copy_advect_gpu2d == null;
        if (!should_reinitialize_temps)
        {
            bool dimention_mismatch =   vx_copy_advect_gpu2d.count != size.x * size.y ||
                                        vy_copy_advect_gpu2d.count != size.x * size.y;

            should_reinitialize_temps = should_reinitialize_temps || dimention_mismatch;
        }

        if (should_reinitialize_temps)
        {
            if (vx_copy_advect_gpu2d != null) vx_copy_advect_gpu2d.Release();
            vx_copy_advect_gpu2d = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);

            if (vy_copy_advect_gpu2d != null) vy_copy_advect_gpu2d.Release();
            vy_copy_advect_gpu2d = new ComputeBuffer(size.x * size.y, sizeof(float), ComputeBufferType.Structured);
        }

        int kernel_index = solver.FindKernel("copy_buffers");
        solver.SetBuffer(kernel_index, "copy_source_buffer", velocX);
        solver.SetBuffer(kernel_index, "copy_target_buffer", vx_copy_advect_gpu2d);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        kernel_index = solver.FindKernel("copy_buffers");
        solver.SetBuffer(kernel_index, "copy_source_buffer", velocY);
        solver.SetBuffer(kernel_index, "copy_target_buffer", vy_copy_advect_gpu2d);
        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / 8.0f), Mathf.CeilToInt(size.y / 8.0f), 1);

        kernel_index = solver.FindKernel("advect");
        solver.SetInt("size_x", size.x);
        solver.SetInt("size_y", size.y);
        solver.SetInt("size_z", size.z);
        solver.SetInt("b", b);
        solver.SetFloat("dt", dt);
        solver.SetBuffer(kernel_index, "d", d);
        solver.SetBuffer(kernel_index, "d0", d0);
        solver.SetBuffer(kernel_index, "velocity_x_1", vx_copy_advect_gpu2d);
        solver.SetBuffer(kernel_index, "velocity_y_1", vy_copy_advect_gpu2d);
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

        add_external_force_gpu3d(gravity.x * 0.0001f, Vx_cb);
        add_external_force_gpu3d(gravity.y * 0.0001f, Vy_cb);
        add_external_force_gpu3d(gravity.z * 0.0001f, Vz_cb);

        diffuse_gpu3d(1, Vx0_cb, Vx_cb);
        diffuse_gpu3d(2, Vy0_cb, Vy_cb);
        diffuse_gpu3d(3, Vz0_cb, Vz_cb);

        project_gpu3d(Vx0_cb, Vy0_cb, Vz0_cb, Vx_cb, Vy_cb);

        advect_gpu3d(1, Vx_cb, Vx0_cb, Vx0_cb, Vy0_cb, Vz0_cb);
        advect_gpu3d(2, Vy_cb, Vy0_cb, Vx0_cb, Vy0_cb, Vz0_cb);
        advect_gpu3d(3, Vz_cb, Vz0_cb, Vx0_cb, Vy0_cb, Vz0_cb);

        project_gpu3d(Vx_cb, Vy_cb, Vz_cb, Vx0_cb, Vy0_cb);

        diffuse_gpu3d(0, s_cb, density_cb);
        advect_gpu3d(0, density_cb, s_cb, Vx_cb, Vy_cb, Vz_cb);

        step_total_cost_3d += Time.realtimeSinceStartup * 1000 - step_begin_time;
    }

    /// <summary>
    /// add density to GPU buffer
    /// </summary>
    private void add_density_gpu3d(int x, int y, int z, float amount)
    {
        init_gpu();

        _write_to_texture_3d(density_cb, x, y, z, amount, true);
    }

    /// <summary>
    /// add velocity to GPU buffer
    /// </summary>
    private void add_velocity_gpu3d(int x, int y, int z, float amount_x, float amount_y, float amount_z)
    {
        init_gpu();

        _write_to_texture_3d(Vx_cb, x, y, z, amount_x, true);
        _write_to_texture_3d(Vy_cb, x, y, z, amount_y, true);
        _write_to_texture_3d(Vz_cb, x, y, z, amount_z, true);
    }

    private void _set_boundry_internal3d(int x, int y, int z, float amount)
    {
        _write_to_texture_3d(boundries_cb, x, y, z, amount);
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
    private void set_bnd_gpu3d(int b, ComputeBuffer x)
    {
        float set_bnd_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();

        int kernel_index = solver3D.FindKernel("set_bnd_3d");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetInt("b", b);
        solver3D.SetBuffer(kernel_index, "x", x);
        solver3D.SetBuffer(kernel_index, "boundries", boundries_cb);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.y / solver3D_kernel_size.z));

        set_bnd_total_cost_3d += Time.realtimeSinceStartup * 1000 - set_bnd_begin_time;
        set_bnd_call_count_3d++;
    }

    /// <summary>
    /// simulate diffusion, GPU mode
    /// </summary>
    private void diffuse_gpu3d(int b, ComputeBuffer x, ComputeBuffer x0)
    {
        init_gpu();

        float a = dt * diff * (size.x - 2) * (size.y - 2);
        lin_solve_gpu3d(b, x, x0, a, 1 + 6 * a);
    }

    // second copy of a texture used for jacobi iterations
    private ComputeBuffer temp_linear_solve_jacobi;

    /// <summary>
    /// solve linear system, GPU mode
    /// </summary>
    private void lin_solve_gpu3d(int b, ComputeBuffer x, ComputeBuffer x0, float a, float c)
    {
        float lin_solve_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();
        if (linear_solver_method == GPULinearSolver.JACOBI)
        {
            if (temp_linear_solve_jacobi == null || temp_linear_solve_jacobi.count != size.x * size.y * size.z)
            {
                if (temp_linear_solve_jacobi != null) temp_linear_solve_jacobi.Release();
                temp_linear_solve_jacobi = new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            }

            int copy_kernel_index = solver3D.FindKernel("copy_buffers_3d");
            solver3D.SetBuffer(copy_kernel_index, "copy_source_buffer", x);
            solver3D.SetBuffer(copy_kernel_index, "copy_target_buffer", temp_linear_solve_jacobi);

            int lin_solve_kernel_index = solver3D.FindKernel("lin_solve_jacobi_3d");
            solver3D.SetInt("size_x", size.x);
            solver3D.SetInt("size_y", size.y);
            solver3D.SetInt("size_z", size.z);
            solver3D.SetInt("b", b);
            solver3D.SetFloat("a", a);
            solver3D.SetFloat("c", c);
            solver3D.SetBuffer(lin_solve_kernel_index, "x", x);
            solver3D.SetBuffer(lin_solve_kernel_index, "x0", x0);
            solver3D.SetBuffer(lin_solve_kernel_index, "x_old", temp_linear_solve_jacobi);

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
            solver3D.SetBuffer(kernel_index, "x", x);
            solver3D.SetBuffer(kernel_index, "x0", x0);
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
    private void add_external_force_gpu3d(float force, ComputeBuffer d)
    {
        init_gpu();

        int kernel_index = solver3D.FindKernel("add_external_force_3d");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetFloat("force", force);
        solver3D.SetFloat("dt", dt);
        solver3D.SetBuffer(kernel_index, "d", d);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        set_bnd_gpu3d(0, d);
    }

    /// <summary>
    /// regulate velocities to achieve zero divergence everywhere, GPU mode
    /// </summary>
    private void project_gpu3d(ComputeBuffer velocX, ComputeBuffer velocY, ComputeBuffer velocZ, ComputeBuffer p, ComputeBuffer div)
    {
        float project_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();

        int kernel_index = solver3D.FindKernel("project_1_3d");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetBuffer(kernel_index, "p", p);
        solver3D.SetBuffer(kernel_index, "div", div);
        solver3D.SetBuffer(kernel_index, "velocity_x_1", velocX);
        solver3D.SetBuffer(kernel_index, "velocity_y_1", velocY);
        solver3D.SetBuffer(kernel_index, "velocity_z_1", velocZ);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        set_bnd_gpu3d(0, div);
        set_bnd_gpu3d(0, p);
        lin_solve_gpu3d(0, p, div, 1, 6);

        kernel_index = solver3D.FindKernel("project_2_3d");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetBuffer(kernel_index, "p", p);
        solver3D.SetBuffer(kernel_index, "div", div);
        solver3D.SetBuffer(kernel_index, "velocity_x_1", velocX);
        solver3D.SetBuffer(kernel_index, "velocity_y_1", velocY);
        solver3D.SetBuffer(kernel_index, "velocity_z_1", velocZ);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        set_bnd_gpu3d(1, velocX);
        set_bnd_gpu3d(2, velocY);
        set_bnd_gpu3d(3, velocZ);

        project_total_cost_3d += Time.realtimeSinceStartup * 1000 - project_begin_time;
        project_call_count_3d++;
    }

    // second copies of velocities for advect
    private ComputeBuffer temp_advect_vx;
    private ComputeBuffer temp_advect_vy;
    private ComputeBuffer temp_advect_vz;

    /// <summary>
    /// flow given field in the direction of velocity fields, GPU mode
    /// </summary>
    private void advect_gpu3d(int b, ComputeBuffer d, ComputeBuffer d0, ComputeBuffer velocX, ComputeBuffer velocY, ComputeBuffer velocZ)
    {
        float advect_begin_time = Time.realtimeSinceStartup * 1000;

        init_gpu();

        bool should_initialize_temps = temp_advect_vx == null || temp_advect_vy == null || temp_advect_vz == null;
        if (!should_initialize_temps)
            should_initialize_temps = should_initialize_temps ||
                            temp_advect_vx.count != size.x * size.y * size.z ||
                            temp_advect_vy.count != size.x * size.y * size.z ||
                            temp_advect_vz.count != size.x * size.y * size.z;

        int kernel_index;

        if (should_initialize_temps)
        {
            if (temp_advect_vx != null) temp_advect_vx.Release();
            if (temp_advect_vy != null) temp_advect_vy.Release();
            if (temp_advect_vz != null) temp_advect_vz.Release();

            temp_advect_vx = new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            temp_advect_vy = new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
            temp_advect_vz = new ComputeBuffer(size.x * size.y * size.z, sizeof(float), ComputeBufferType.Structured);
        }

        float texture_copies_begin_time = Time.realtimeSinceStartup * 1000;

        kernel_index = solver3D.FindKernel("copy_buffers_3d");
        solver3D.SetBuffer(kernel_index, "copy_source_buffer", velocX);
        solver3D.SetBuffer(kernel_index, "copy_target_buffer", temp_advect_vx);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        kernel_index = solver3D.FindKernel("copy_buffers_3d");
        solver3D.SetBuffer(kernel_index, "copy_source_buffer", velocY);
        solver3D.SetBuffer(kernel_index, "copy_target_buffer", temp_advect_vy);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        kernel_index = solver3D.FindKernel("copy_buffers_3d");
        solver3D.SetBuffer(kernel_index, "copy_source_buffer", velocZ);
        solver3D.SetBuffer(kernel_index, "copy_target_buffer", temp_advect_vz);
        solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), Mathf.CeilToInt(size.z / solver3D_kernel_size.z));

        copy_textures_total_cost_3d += Time.realtimeSinceStartup * 1000 - texture_copies_begin_time;
        copy_textures_call_count_3d += 3;

        kernel_index = solver3D.FindKernel("advect_3d");
        solver3D.SetInt("size_x", size.x);
        solver3D.SetInt("size_y", size.y);
        solver3D.SetInt("size_z", size.z);
        solver3D.SetInt("b", b);
        solver3D.SetFloat("dt", dt);
        solver3D.SetBuffer(kernel_index, "d", d);
        solver3D.SetBuffer(kernel_index, "d0", d0);
        solver3D.SetBuffer(kernel_index, "velocity_x_1", temp_advect_vx);
        solver3D.SetBuffer(kernel_index, "velocity_y_1", temp_advect_vy);
        solver3D.SetBuffer(kernel_index, "velocity_z_1", temp_advect_vz);
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
        //if (mode == ComputationMode.GPU3D) set_boundry_gpu3d(x0, y0, z0, size_x, size_y, size_z, is_boundry);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_density(int x, int y, int z, float amount)
    {
        if (mode == ComputationMode.GPU2D) add_density_gpu(x, y, amount);
        //if (mode == ComputationMode.GPU3D) add_density_gpu3d(x, y, z, amount);
    }

    /// <summary>
    /// add density to current mode's buffer
    /// </summary>
    public void add_velocity(int x, int y, int z, float amount_x, float amount_y, float amount_z)
    {
        if (mode == ComputationMode.GPU2D) add_velocity_gpu(x, y, amount_x, amount_y);
        //if (mode == ComputationMode.GPU3D) add_velocity_gpu3d(x, y, z, amount_x, amount_y, amount_z);
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
            else if(mode == ComputationMode.GPU2D) step_gpu();
        }
    }

    public Vector3Int simulation_resolution;
    public int simulation_iteration_per_step;
    public float diffusion;

    public int wall_thickness;
    void Start()
    {
        init_parameters(simulation_resolution, simulation_iteration_per_step, 0.1f, 0.0001f * diffusion, 0.1f);

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
    //private void OnRenderImage(RenderTexture source, RenderTexture destination)
    //{
    //    if (_first_draw_in_gpu_mode)
    //    {
    //        init_gpu();
    //        if (slice_of_3D) slice_of_3D.Release();
    //        slice_of_3D = new RenderTexture(size.x, size.y, 0);
    //        slice_of_3D.filterMode = FilterMode.Bilinear;
    //        slice_of_3D.enableRandomWrite = true;
    //    }
    //    _first_draw_in_gpu_mode = false;
    //
    //    if (mode == ComputationMode.GPU2D)
    //    {
    //        int kernel_index = solver.FindKernel("copy_to_texture");
    //        solver.SetBuffer(kernel_index, "copy_source_buffer", density_cb);
    //        solver.SetTexture(kernel_index, "copy_target_texture", slice_of_3D);
    //        solver.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), 1);
    //        Graphics.Blit(slice_of_3D, (RenderTexture)null);
    //    }
    //
    //    //if (mode == ComputationMode.GPU3D && _gpu_buffers_sized_for_3D)
    //    //{
    //    //    int kernel_index = solver3D.FindKernel("get_3d_texture_slice");
    //    //    solver3D.SetInt("slice_z_index", displaying_slice_index);
    //    //    solver3D.SetTexture(kernel_index, "copy_source_texture", density_rt);
    //    //    solver3D.SetTexture(kernel_index, "slice_target", slice_of_3D);
    //    //    solver3D.Dispatch(kernel_index, Mathf.CeilToInt(size.x / solver3D_kernel_size.x), Mathf.CeilToInt(size.y / solver3D_kernel_size.y), 1);
    //    //    Graphics.Blit(slice_of_3D, (RenderTexture)null);
    //    //}
    //}
}
