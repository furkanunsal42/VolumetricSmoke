using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VolumetricSmoke : MonoBehaviour
{
    public ComputeShader volumetric_smoke_compute_shader;

    public Vector3 worley_noise_size;
    public Vector3 worley_noise_division_size;
    public ComputeBuffer worley_points_buffer;
    public RenderTexture worley_noise_texture;

    public RenderTexture detail_noise;

    public Material volume_shader_material;

    private bool _compute_noise_2D_first_execution = true;
    private bool _compute_noise_3D_first_execution = true;
    private bool _light_map_3D_initialized = false;

    private bool _point_buffer_2D_first_execution = true;
    private bool _point_buffer_3D_first_execution = true;

    public int displaying_3d_slice = 0;
    public bool iterate_over_z = false;

    private RenderTexture noise_slice_texture;  // for debug

    [Range(0, 1)]
    public float density_threshold;

    [Range(0, 100)]
    public float light_decay;

    [Range(0, 0.03f)]
    public float god_rays_strength;

    [Range(1, 512)]
    public int density_sample_count;

    [Range(0, 128)]
    public int light_sample_count;

    public Vector3 light_color;
    
    public Vector3 light_direction;

    // lighting computations
    public bool use_light_map;
    public RenderTexture light_map_3D;
    public Vector3 light_map_resolution;

    public Camera scene_camera;
    
    private void init_parameters()
    {
        _compute_noise_2D_first_execution = true;
        _compute_noise_3D_first_execution = true;
        _light_map_3D_initialized = false;

        _point_buffer_2D_first_execution = true;
        _point_buffer_3D_first_execution = true;

        displaying_3d_slice = 0;
        iterate_over_z = false;

        iterate_z_every = 1;
        iterate_z_counter = 0;

        update_light_map_every = 1;
        update_light_map_counter = 0;
    }

    private void _init_2D()
    {
        int count = (int)worley_noise_size.x / (int)worley_noise_division_size.x * (int)worley_noise_size.y / (int)worley_noise_division_size.y;

        if (_compute_noise_2D_first_execution)
        {
            _compute_noise_2D_first_execution = false;
            _compute_noise_3D_first_execution = true;


            worley_noise_texture = new RenderTexture((int)worley_noise_size.x, (int)worley_noise_size.y, 0);
            worley_noise_texture.enableRandomWrite = true;
            worley_noise_texture.wrapMode = TextureWrapMode.Repeat;
            worley_noise_texture.filterMode = FilterMode.Bilinear;
            worley_noise_texture.format = RenderTextureFormat.ARGB32;
        }

        if (true /*_point_buffer_2D_first_execution*/)
        {
            _point_buffer_2D_first_execution = false;
            _point_buffer_3D_first_execution = true;

            float[] points = new float[count * 2];

            for (int i = 0; i < count; i++)
            {
                points[2 * i + 0] = Random.value;
                points[2 * i + 1] = Random.value;
            }

            if (worley_points_buffer != null)
                worley_points_buffer.Dispose();
            worley_points_buffer = new ComputeBuffer(count, sizeof(float) * 2, ComputeBufferType.Structured);
            worley_points_buffer.SetData(points);
        }
    }

    private void _compute_worley_noise_2D(float strength = 1.0f)
    {
        _init_2D();

        int kernel_index;
        kernel_index = volumetric_smoke_compute_shader.FindKernel("add_worley_noise_2D");
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_x", (int)worley_noise_size.x);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_y", (int)worley_noise_size.y);
        volumetric_smoke_compute_shader.SetInt("cell_size_x", (int)worley_noise_division_size.x);
        volumetric_smoke_compute_shader.SetInt("cell_size_y", (int)worley_noise_division_size.y);
        volumetric_smoke_compute_shader.SetFloat("noise_strength", strength);
        volumetric_smoke_compute_shader.SetBuffer(kernel_index, "feature_points_buffer_2D", worley_points_buffer);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "noise_texture_2D", worley_noise_texture);

        volumetric_smoke_compute_shader.Dispatch(kernel_index, Mathf.CeilToInt(worley_noise_size.x / 8), Mathf.CeilToInt(worley_noise_size.y / 8), 1);

        //worley_points_buffer.Dispose();
    }

    private void _init_light_map_3D()
    {
        if (_light_map_3D_initialized) return;
        _light_map_3D_initialized = true;

        light_map_3D = new RenderTexture((int)light_map_resolution.x, (int)light_map_resolution.y, 0)
        {
            enableRandomWrite = true,
            dimension = UnityEngine.Rendering.TextureDimension.Tex3D,
            volumeDepth = (int)light_map_resolution.z,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
            format = RenderTextureFormat.R8
        };

        volume_shader_material.SetTexture("_LightTexture", light_map_3D);

    }

    private void _init_3D()
    {
        int cell_count = Mathf.CeilToInt((worley_noise_size.x / worley_noise_division_size.x) * (worley_noise_size.y / worley_noise_division_size.y) * (worley_noise_size.z / worley_noise_division_size.z));

        if (_compute_noise_3D_first_execution)
        {
            _compute_noise_2D_first_execution = true;
            _compute_noise_3D_first_execution = false;

            worley_noise_texture = new RenderTexture((int)worley_noise_size.x, (int)worley_noise_size.y, 0)
            {
                enableRandomWrite = true,
                dimension = UnityEngine.Rendering.TextureDimension.Tex3D,
                volumeDepth = (int)worley_noise_size.z,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
                format = RenderTextureFormat.R8
            };

            volume_shader_material.SetTexture("_DensityTexture", worley_noise_texture);

            detail_noise = new RenderTexture((int)worley_noise_size.x, (int)worley_noise_size.y, 0)
            {
                enableRandomWrite = true,
                dimension = UnityEngine.Rendering.TextureDimension.Tex3D,
                volumeDepth = (int)worley_noise_size.z,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
                format = RenderTextureFormat.R8
            };

            volume_shader_material.SetTexture("_DensityDetailTexture", detail_noise);


            _init_light_map_3D();
        }

        bool point_buffer_size_changed = false;
        if (worley_points_buffer != null)
            point_buffer_size_changed = worley_points_buffer.count != cell_count;

        if (true /*_point_buffer_3D_first_execution || point_buffer_size_changed*/)
        {
            _point_buffer_2D_first_execution = true;
            _point_buffer_3D_first_execution = false;

            if (worley_points_buffer != null)
                worley_points_buffer.Dispose();

            worley_points_buffer = new ComputeBuffer(cell_count, sizeof(float) * 3, ComputeBufferType.Structured);
            float[] points = new float[cell_count * 3];
            for (int i = 0; i < cell_count; i++)
            {
                points[3 * i + 0] = Random.value;
                points[3 * i + 1] = Random.value;
                points[3 * i + 2] = Random.value;
            }

            worley_points_buffer.SetData(points);
        }
    }

    private void _compute_worley_noise_3D(float strength = 1.0f)
    {
        _init_3D();

        int kernel_index;
        kernel_index = volumetric_smoke_compute_shader.FindKernel("add_worley_noise_3D");
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_x", (int)worley_noise_size.x);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_y", (int)worley_noise_size.y);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_z", (int)worley_noise_size.z);
        volumetric_smoke_compute_shader.SetInt("cell_size_x", (int)worley_noise_division_size.x);
        volumetric_smoke_compute_shader.SetInt("cell_size_y", (int)worley_noise_division_size.y);
        volumetric_smoke_compute_shader.SetInt("cell_size_z", (int)worley_noise_division_size.z);
        volumetric_smoke_compute_shader.SetFloat("noise_strength", strength);
        volumetric_smoke_compute_shader.SetBuffer(kernel_index, "feature_points_buffer_3D", worley_points_buffer);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "noise_texture_3D", worley_noise_texture);

        volumetric_smoke_compute_shader.Dispatch(kernel_index, Mathf.CeilToInt(worley_noise_size.x / 8), Mathf.CeilToInt(worley_noise_size.y / 8), Mathf.CeilToInt(worley_noise_size.z / 8));

        //worley_points_buffer.Dispose();
    }
    private void _compute_worley_noise_3D_detail(float strength = 1.0f)
    {
        _init_3D();

        int kernel_index;
        kernel_index = volumetric_smoke_compute_shader.FindKernel("add_worley_noise_3D");
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_x", (int)worley_noise_size.x);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_y", (int)worley_noise_size.y);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_z", (int)worley_noise_size.z);
        volumetric_smoke_compute_shader.SetInt("cell_size_x", (int)worley_noise_division_size.x);
        volumetric_smoke_compute_shader.SetInt("cell_size_y", (int)worley_noise_division_size.y);
        volumetric_smoke_compute_shader.SetInt("cell_size_z", (int)worley_noise_division_size.z);
        volumetric_smoke_compute_shader.SetFloat("noise_strength", strength);
        volumetric_smoke_compute_shader.SetBuffer(kernel_index, "feature_points_buffer_3D", worley_points_buffer);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "noise_texture_3D", detail_noise);

        volumetric_smoke_compute_shader.Dispatch(kernel_index, Mathf.CeilToInt(worley_noise_size.x / 8), Mathf.CeilToInt(worley_noise_size.y / 8), Mathf.CeilToInt(worley_noise_size.z / 8));
    }

    public Texture2D white_noise;
    private void _generate_white_noise_cpu()
    {
        int size = 64 * 64;
        Color[] texture_data = new Color[size];
        for(int i = 0; i < size; i++)
            texture_data[i].r = Random.value;

        if (white_noise != null)
            DestroyImmediate(white_noise);
        
        white_noise = new Texture2D(64, 64, TextureFormat.R16, false);
        white_noise.wrapMode = TextureWrapMode.Mirror;

        white_noise.SetPixels(texture_data);
        white_noise.Apply();

        volume_shader_material.SetTexture("_WhiteNoise", white_noise);
    }

    private void _compute_light_map()
    {
        _init_light_map_3D();
        
        if (solver == null) return;

        int kernel_index;
        kernel_index = volumetric_smoke_compute_shader.FindKernel("compute_light_intensities");

        volumetric_smoke_compute_shader.SetInt("noise_texture_size_x", (int)worley_noise_size.x);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_y", (int)worley_noise_size.y);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_z", (int)worley_noise_size.z);

        volumetric_smoke_compute_shader.SetVector("light_texture_dims", light_map_resolution);
        volumetric_smoke_compute_shader.SetVector("global_light_direction", light_direction);
        volumetric_smoke_compute_shader.SetVector("global_light_color", light_color);

        volumetric_smoke_compute_shader.SetFloat("light_decay", light_decay);
        volumetric_smoke_compute_shader.SetFloat("god_rays_intensity", god_rays_strength);
        volumetric_smoke_compute_shader.SetFloat("step_size", 1.0f / light_sample_count);
        volumetric_smoke_compute_shader.SetFloat("density_threashold", density_threshold);

        volumetric_smoke_compute_shader.SetTexture(kernel_index, "light_density_texture", light_map_3D);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "noise_texture_3D", worley_noise_texture);
        
        volumetric_smoke_compute_shader.SetVector("mask_texture_dims", new Vector3(solver.density_rt.width, solver.density_rt.height, solver.density_rt.volumeDepth));
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "mask_texture_3D", solver.density_rt);

        volumetric_smoke_compute_shader.Dispatch(kernel_index, Mathf.CeilToInt(light_map_resolution.x / 8), Mathf.CeilToInt(light_map_resolution.y / 8), Mathf.CeilToInt(light_map_resolution.z / 8));
    }

    private void generate_textures()
    {
        _generate_white_noise_cpu();

        _compute_worley_noise_3D(0.4f);
        worley_noise_division_size /= 2;
        _compute_worley_noise_3D(0.2f);
        worley_noise_division_size /= 2;
        _compute_worley_noise_3D(0.1f);
        _compute_worley_noise_3D(0.1f);
        _compute_worley_noise_3D(0.1f);
        _compute_worley_noise_3D(0.1f);

        int octave_count = 5;
        float persistance = 0.5f;
        float lacurancy = 2.0f;

        float amplitude = 0.075f;
        int size = 16;
        for (int octave = 0; octave < octave_count; octave++)
        {
            worley_noise_division_size = new Vector3(size, size, size);
            _compute_worley_noise_3D_detail(amplitude);
            amplitude *= persistance;
            size = Mathf.CeilToInt(size / lacurancy);
        }
    }

    RenderTexture union_texture;

    private void merge_all_textures()
    {
        if (solver == null) return;
        
        int max_size = Mathf.Max(new int[] { (int)worley_noise_size.x, (int)light_map_resolution.x, solver.density_rt.width });
        if (union_texture == null)
        {
            union_texture = new RenderTexture(max_size, max_size, 0);
            union_texture.graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_UNorm;
            union_texture.volumeDepth = max_size;
            union_texture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
            union_texture.filterMode = FilterMode.Bilinear;
            union_texture.enableRandomWrite = true;
        }

        int kernel_index = volumetric_smoke_compute_shader.FindKernel("unite_textures");
        volumetric_smoke_compute_shader.SetInt("united_texture_size"    , union_texture.width);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size"     , worley_noise_texture.width);
        volumetric_smoke_compute_shader.SetInt("detail_texture_size"    , detail_noise.width);
        volumetric_smoke_compute_shader.SetInt("light_texture_size"     , light_map_3D.width);
        volumetric_smoke_compute_shader.SetInt("mask_texture_size"      , solver.density_rt.width);

        volumetric_smoke_compute_shader.SetTexture(kernel_index, "united_texture"       , union_texture);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "noise_texture_merge"  , worley_noise_texture);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "detail_texture_merge" , detail_noise);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "light_texture_merge"  , light_map_3D);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "mask_texture_merge"   , solver.density_rt);

        volumetric_smoke_compute_shader.Dispatch(kernel_index, Mathf.CeilToInt(max_size / 8.0f), Mathf.CeilToInt(max_size / 8.0f), Mathf.CeilToInt(max_size / 8.0f));

        volume_shader_material.SetTexture("_UnitedTexture", solver.density_rt);
    }

    public NavierStokesSolver solver;
    public SimpleSmokePhysics simple_solver;

    private void set_camera_deferred()
    {
        //ContentManager.Instance.OpenXRMainCamera.GetComponent<Camera>().renderingPath = RenderingPath.DeferredShading;
        //ContentManager.Instance.OpenXRMainCamera.GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;
        //ContentManager.Instance.OpenXRMainCamera.GetComponent<Camera>().clearFlags = CameraClearFlags.Color;
        //
        //ContentManager.Instance.CameraOVRLeft.GetComponent<Camera>().renderingPath = RenderingPath.DeferredShading;
        //ContentManager.Instance.CameraOVRLeft.GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth;
        //ContentManager.Instance.CameraOVRLeft.GetComponent<Camera>().clearFlags = CameraClearFlags.Color;
        //
        //ContentManager.Instance.CameraOVRRight.GetComponent<Camera>().renderingPath = RenderingPath.DeferredShading;
        //ContentManager.Instance.CameraOVRRight.GetComponent<Camera>().depthTextureMode = DepthTextureMode.Depth;
        //ContentManager.Instance.CameraOVRRight.GetComponent<Camera>().clearFlags = CameraClearFlags.Color;
        //
        //ContentManager.Instance.NoneMainCamera.GetComponentInChildren<Camera>().renderingPath = RenderingPath.DeferredShading;
        //ContentManager.Instance.NoneMainCamera.GetComponentInChildren<Camera>().depthTextureMode = DepthTextureMode.Depth;
        //ContentManager.Instance.NoneMainCamera.GetComponentInChildren<Camera>().clearFlags = CameraClearFlags.Color;
    }

    private void free_vram()
    {
        if (worley_points_buffer != null)   worley_points_buffer.Release();
        if (worley_noise_texture != null)   worley_noise_texture.Release();
        if (detail_noise != null)           detail_noise.Release();
        if (noise_slice_texture != null)    noise_slice_texture.Release();
        if (light_map_3D != null)           light_map_3D.Release();
        if (union_texture != null)          union_texture.Release();
    }

    public void OnDestroy()
    {
        free_vram();
    }

    void Start()
    {
        scene_camera.renderingPath = RenderingPath.DeferredShading;
        scene_camera.depthTextureMode = DepthTextureMode.Depth;

        Random.InitState((int)(Time.time * 1000000));
        
        generate_textures();
    }

    public bool iterate_time;
    public void Update()
    {
        volume_shader_material.SetInt("use_united_texture", (solver == null) ? 0 : 0);
        volume_shader_material.SetInt("use_light_map", use_light_map ? 1 : 0);
        if (iterate_time)
            solver.step(Time.deltaTime * 1000);
    }

    private int iterate_z_every = 1;
    private int iterate_z_counter = 0;

    private int update_light_map_every = 1;
    private int update_light_map_counter = 0;

    [Range(0, 128)]
    private float time_ms;

    MeshRenderer mesh_renderer_referance;
    private void FixedUpdate()
    {
        if (mesh_renderer_referance == null)
            if (TryGetComponent<MeshRenderer>(out mesh_renderer_referance))
                mesh_renderer_referance.material = volume_shader_material;

        if (solver != null)
        {
            //merge_all_textures();
            volume_shader_material.SetTexture("_MaskTexture", solver.density_rt);
        }

        if (--update_light_map_counter <= 0 && use_light_map)
        {
            _compute_light_map();
            update_light_map_counter = update_light_map_every;
        }

        volume_shader_material.SetVector("object_scale", new Vector4(transform.lossyScale.x, transform.lossyScale.y, transform.lossyScale.z, 1));

        volume_shader_material.SetFloat("density_sample_count", density_sample_count);
        volume_shader_material.SetFloat("light_sample_count", light_sample_count);
        
        volume_shader_material.SetVector("light_color", light_color);
        volume_shader_material.SetVector("light_direction", light_direction);

        volume_shader_material.SetFloat("density_threshold", density_threshold);
        volume_shader_material.SetFloat("light_decay", light_decay);
        volume_shader_material.SetFloat("god_rays_strength", god_rays_strength);
        volume_shader_material.SetVector("texture_offset", new Vector3(0, (-time_ms * 0.003f * worley_noise_size.y % worley_noise_size.y) , 0));
        volume_shader_material.SetVector("noise_texture_size", worley_noise_size);
        volume_shader_material.SetVector("light_texture_size", light_map_resolution);
        if (solver != null && solver.density_rt != null)
            volume_shader_material.SetVector("mask_texture_size", new Vector3(solver.density_rt.width, solver.density_rt.height, solver.density_rt.volumeDepth));

        if (iterate_z_counter-- < 0)
        {
            if (iterate_over_z)
                displaying_3d_slice = ++displaying_3d_slice % (int)worley_noise_size.z;
            iterate_z_counter = iterate_z_every;
        }
    }

    private bool _first_camera_blit = true;
    private void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        if (_first_camera_blit)
        {
            noise_slice_texture = new RenderTexture((int)worley_noise_size.x, (int)worley_noise_size.y, 0);
            noise_slice_texture.enableRandomWrite = true;
            _first_camera_blit = false;
        }

        if (worley_noise_texture == null) return;
        if (worley_points_buffer == null) return;

        int kernel_index;
        kernel_index = volumetric_smoke_compute_shader.FindKernel("get_slice_of_3D_texture");
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_x", (int)worley_noise_size.x);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_y", (int)worley_noise_size.y);
        volumetric_smoke_compute_shader.SetInt("noise_texture_size_z", (int)worley_noise_size.z);
        volumetric_smoke_compute_shader.SetInt("slice_index", displaying_3d_slice);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "noise_texture_3D", worley_noise_texture);
        volumetric_smoke_compute_shader.SetTexture(kernel_index, "slice_texture", noise_slice_texture);
        volumetric_smoke_compute_shader.Dispatch(kernel_index, Mathf.CeilToInt(worley_noise_size.x / 8), Mathf.CeilToInt(worley_noise_size.y / 8), 1);

        Graphics.Blit(noise_slice_texture, (RenderTexture)null);

    }
}
