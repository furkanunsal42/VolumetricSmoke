Shader "Unlit/VolumetricSmokeShader"
{
    Properties
    {
        _DensityTexture("Density Texture", 3D) = "black"
        _DensityDetailTexture("Density Detail Texture", 3D) = "white"
        _MaskTexture("Mask Texture", 3D) = "white"
        _LightTexture("Light Texture", 3D) = "white"
        _WhiteNoise("White Noise", 2D) = "black"
        _UnitedTexture("United Texture", 3D) = "black"
    }
        SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
        LOD 100
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Front
        ZWrite On
        ZTest Off


        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"


            struct appdata
            {
                float4 vPos : POSITION;
                float3 worldPos : TEXCOORD1;
            };

            struct v2f
            {
                float4 screenPos : SV_POSITION;
                float3 worldPos : TEXCOORD1;
                float3 objectPos : TEXCOORD3;

                float3 cameraViewDir : TEXCOORD2;
            };

            float3 texture_offset;
            float3 noise_texture_size;
            float3 light_texture_size;
            float3 mask_texture_size;

            sampler3D _DensityTexture;
            sampler3D _DensityDetailTexture;
            sampler3D_float _MaskTexture;
            sampler3D _LightTexture;
            sampler2D _WhiteNoise;
            sampler3D _UnitedTexture;

            sampler2D _CameraDepthTexture;  // Unity

            struct MergedSample {
                float noise;
                float detail;
                float light;
                float mask;
            };

            float3 united_texture_size;
            float3 last_sampled_texcoord;
            
            int use_united_texture;
            int use_light_map;

            float4 object_scale;

            MergedSample sample_merged_texture(float3 texCoord) {
                
                last_sampled_texcoord = texCoord;
                MergedSample sample_data;

                if (!use_united_texture) {
                    
                    float3 minimalized_object_scale = object_scale.xyz / max(max(object_scale.x, object_scale.y), object_scale.z);
                    
                    last_sampled_texcoord *= minimalized_object_scale;

                    sample_data.noise = tex3D(_DensityTexture, last_sampled_texcoord + 0.5).r;
                    sample_data.detail = tex3D(_DensityDetailTexture, last_sampled_texcoord + 0.5).r;
                    
                    if (use_light_map)
                        sample_data.light = tex3D(_LightTexture, last_sampled_texcoord + 0.5).r;
                    else
                        sample_data.light = 0.65f;

                    last_sampled_texcoord /= minimalized_object_scale;
                    
                    sample_data.mask = tex3D(_MaskTexture, texCoord + 0.5).r;
                    
                    return sample_data;
                }
                else {
                    float4 data = tex3D(_UnitedTexture, fmod(texCoord + 0.5 + texture_offset / noise_texture_size + 1, 1));

                    sample_data.noise = data.r;
                    sample_data.detail = data.g;
                    sample_data.light = data.b;
                    sample_data.mask = data.a * 32.0f;

                    return sample_data;
                }
                

                //sample_data.noise = data.r;
                //sample_data.detail = data.g;
                //sample_data.light = data.b;
                //sample_data.mask = data.a * 32.0f;

                
            }

            float sample_white_noise(float2 texCoord) {
                return tex2D(_WhiteNoise, fmod(float2(texCoord.x + texCoord.y, texCoord.x - texCoord.y), 1)).r;
            }

            float sample_light_intensity(MergedSample sample_data) {
                return sample_data.light;

            }
            
            float density_threshold;

            float sample_density(MergedSample sample_data) {
                float density;
                float detail;
                float mask;

                //if (!use_united_texture) {
                //    density = tex3D(_DensityTexture, fmod(last_sampled_texcoord + 0.5 + texture_offset / noise_texture_size + 1 /*+ float3(0, _Time.x.x, 0)*/, 1)).r;
                //    detail = tex3D(_DensityDetailTexture, fmod(1 + fmod(last_sampled_texcoord + 0.5 + texture_offset / noise_texture_size + float3(0, -_Time.x, 0) / 8, 1), 1)).r;
                //    mask = 0.2;
                //}
                //else {
                    density = sample_data.noise;
                    detail = sample_data.detail;
                    mask = sample_data.mask;
                //}
                    
                //density += detail;

                //float margin = 1;
                //if (texCoord.x > margin || texCoord.y > margin || texCoord.z > margin) mask = 0;
                //if (texCoord.x < -margin || texCoord.y < -margin || texCoord.z < -margin) mask = 0;

                float masked_density = min(1, abs(mask - density_threshold)) * (detail * 10 * 2);
                return masked_density;
            }

            //Color blending (front-to-back)
            float4 BlendFTB(float4 color, float4 newColor)
            {
                color.rgb += (1.0 - color.a) * newColor.a * newColor.rgb;
                color.a += (1.0 - color.a) * newColor.a;
                return color;
            }

            //Color blending (back-to-front)
            float4 BlendBFT(float4 color, float4 newColor)
            {
                return (1 - newColor.a) * color + newColor.a * newColor;
            }


            v2f vert(appdata v)
            {
                v2f output;
                output.objectPos = v.vPos.xyz;
                output.worldPos = mul(unity_ObjectToWorld, v.vPos).xyz;
                output.screenPos = UnityObjectToClipPos(v.vPos);

                output.cameraViewDir = mul(UNITY_MATRIX_MV, float4(v.vPos.xyz, 1.0f)).xyz;

                return output;
            }

            float light_decay;
            float god_rays_strength;

            int density_sample_count;
            int light_sample_count;
            int light_sample_per_density_sample;

            float3 light_color;
            float3 light_direction;

            //It controls given ray intersects with box (x: distance to box, y: distance inside box)
            float2 RayIntersectAABB(float3 boundsMin, float3 boundsMax, float3 rayOrigin, float3 rayDir) {

                float3 invRayDir = 1.0f / rayDir;

                float3 t0 = (boundsMin - rayOrigin) * invRayDir;
                float3 t1 = (boundsMax - rayOrigin) * invRayDir;
                float3 tmin = min(t0, t1);
                float3 tmax = max(t0, t1);

                float dstA = max(max(tmin.x, tmin.y), tmin.z);
                float dstB = min(tmax.x, min(tmax.y, tmax.z));

                float dstToBox = max(0, dstA);
                float dstInsideBox = max(0, dstB - dstToBox);

                return float2(dstToBox, dstInsideBox);
            }

            //float sample_ray_light(float3 start_position, float3 view_direction, float distance, int step_count = 8) {
            //    if (step_count == 0)
            //        return exp(-distance * light_decay);
            //
            //    float final_intensity = 1;
            //    float step_size = distance / (float)step_count;
            //    for (int i = 0; i < step_count; i++) {
            //        float3 sample_location = start_position + view_direction * step_size * i;
            //        float density = sample_density(/*sample_location*/);
            //        if (density < density_threshold)
            //            density = 0.0;
            //        final_intensity *= exp(-density * step_size * light_decay);
            //    }
            //    return final_intensity;
            //}

            float4 sample_ray(float3 start_position, float3 view_direction, float distance, int step_count, float2 uv) {
                float4 final_color = float4(0, 0, 0, 0);
                float step_size = distance / step_count;

                float random_margin = sample_white_noise(uv.xy / 100) * view_direction * step_size * 1;
                start_position += random_margin;

                float empty_step_speed = 1;
                float smoke_step_speed = 1;

                float current_speed = smoke_step_speed;

                float travelled_distance = 0.01;
                [loop]
                while (travelled_distance < distance) {
                    float3 sample_position = start_position + view_direction * travelled_distance;
                    MergedSample sample_data = sample_merged_texture(sample_position);

                    float density = sample_density(sample_data);
                    
                    float light_intensity = sample_light_intensity(sample_data);

                    if (density < density_threshold)
                        density = god_rays_strength;

                    if (density < god_rays_strength + 0.01f && light_intensity > 0.5f)
                        light_intensity *= 1.5;

                    float3 current_light_color = light_color * light_intensity;

                    //if (light_intensity > 0.9) {
                    //    light_color = (light_color * 0.9f + light_intensity * float3(0.3, 0.8, 0.9) * 0.1);
                    //}

                    final_color = BlendFTB(final_color, float4(current_light_color.xyz, density * step_size * current_speed * 24));
                  
                    if (final_color.a >= 0.96f)
                        break;

                    travelled_distance += step_size * current_speed;

                    if (density < density_threshold)    current_speed = empty_step_speed;
                    else                                current_speed = smoke_step_speed;

                    //if (final_color.a >= 0.96f) return final_color;
                }

                //[loop]
                //for (int i = 0; i < step_count - 1; i++) {
                //    float3 sample_location = start_position + view_direction * step_size * i;
                //    float density = sample_density(sample_location);
                //
                //    float light_intensity = sample_light_intensity(sample_location);
                //
                //    if (density < density_threshold)
                //        density = god_rays_strength;
                //
                //
                //    if (density < god_rays_strength + 0.01f && light_intensity > 0.5f)
                //        light_intensity *= 1.5;
                //
                //    float3 current_light_color = light_color * light_intensity;
                //
                //    //if (light_intensity > 0.9) {
                //    //    light_color = (light_color * 0.9f + light_intensity * float3(0.3, 0.8, 0.9) * 0.1);
                //    //}
                //
                //    final_color = BlendFTB(final_color, float4(current_light_color.xyz, density * step_size * 24));
                //}
                return final_color;
            }

            fixed4 frag(v2f i) : SV_Target
            {

                float3 cameraPos = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1.0f)).xyz; //camera position locale to volume 
                float3 viewDirection = mul(unity_WorldToObject, float4(i.worldPos - (_WorldSpaceCameraPos), 0.0f)).xyz;
                viewDirection = normalize(viewDirection);

                float2 hit = RayIntersectAABB(-0.5, 0.5, cameraPos, viewDirection);

                // depth reading

                float2 screen_uv = i.screenPos.xy / _ScreenParams.xy;
                float depth = tex2D(_CameraDepthTexture, screen_uv).r;
                depth = LinearEyeDepth(depth);

                float cosAngle = dot(normalize(i.cameraViewDir), float3(0, 0, -1));
                depth = depth * (1.0f / cosAngle);

                // transform depth to object space
                depth = depth / object_scale;

                if (hit.x + hit.y > depth) {
                    hit.y = depth - hit.x;
                }

                float3 ray_start_point = cameraPos + viewDirection * (hit.x);
                float4 surface_color = sample_ray(ray_start_point, viewDirection, hit.y, density_sample_count, i.screenPos);

                //if (surface_color.a >= 0.90) surface_color.a = 1;
                
                fixed4 col = fixed4(surface_color.xyz, min(surface_color.a, 1));
                col.xyz = col / col.a;

                //if (col.a < 0.05)
                //    clip(-1);

                return col;
            }
            ENDCG
        }
    }
        Fallback Off
}
