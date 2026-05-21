import taichi as ti
import random

# 初始化 Taichi GPU 后端
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
samples_per_pixel = ti.field(ti.i32, shape=()) # 动态控制抗锯齿采样数

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2  # 新增玻璃材质

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, etai_over_etat):
    """
    计算折射光线方向 (Snell's Law)
    """
    cos_theta = ti.max(-1.0, ti.min(1.0, I.dot(N)))
    out_perp = etai_over_etat * (I - cos_theta * N)
    out_parallel = -ti.sqrt(ti.abs(1.0 - out_perp.norm_sqr())) * N
    return out_perp + out_parallel

@ti.func
def schlick_fresnel(cos_theta, r0):
    """Schlick近似计算反射概率"""
    return r0 + (1.0 - r0) * ti.pow(1.0 - cos_theta, 5.0)

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
        else:
            t2 = (-b + ti.sqrt(delta)) / 2.0
            if t2 > 0:
                t = t2
                p = ro + rd * t
                normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测左侧球 -> 改为半透明玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.4, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([1.0, 1.0, 1.0]) # 玻璃本身的吸收色（白色表示纯净）
        hit_mat = MAT_GLASS

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.4, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 1.5
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (int(ix) + int(iz)) % 2 == 0:
            hit_c = ti.Vector([0.2, 0.2, 0.2])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.25])
    
    # 玻璃材质的折射率 (空气约 1.0, 玻璃约 1.5)
    ior_glass = 1.5 

    for i, j in pixels:
        pixel_color_acc = ti.Vector([0.0, 0.0, 0.0])
        samples = samples_per_pixel[None]
        
        # --- 抗锯齿超采样循环 ---
        for s in range(samples):
            # 给像素坐标施加 [-0.5, 0.5] 的随机抖动偏移
            offset_x = 0.0 if samples == 1 else ti.random() - 0.5
            offset_y = 0.0 if samples == 1 else ti.random() - 0.5
            
            u = ((i + offset_x) - res_x / 2.0) / res_y * 2.0
            v = ((j + offset_y) - res_y / 2.0) / res_y * 2.0
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.1, -1.0]))

            final_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            
            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)
                
                if t > 1e9:
                    final_color += throughput * bg_color
                    break
                    
                p = ro + rd * t
                
                if mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color 
                    
                elif mat_id == MAT_GLASS:
                    # 判断是从外部射入还是内部射出
                    front_face = rd.dot(N) < 0
                    out_normal = N if front_face else -N
                    etai_over_etat = (1.0 / ior_glass) if front_face else ior_glass
                    
                    cos_theta = ti.min(-rd.dot(out_normal), 1.0)
                    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
                    
                    # 检查是否发生全反射 (Total Internal Reflection)
                    cannot_refract = etai_over_etat * sin_theta > 1.0
                    
                    # 计算基础菲涅尔反射率
                    r0 = (1.0 - ior_glass) / (1.0 + ior_glass)
                    r0 = r0 * r0
                    reflect_prob = schlick_fresnel(cos_theta, r0)
                    
                    # 决策：反射还是折射
                    if cannot_refract or ti.random() < reflect_prob:
                        # 全反射或菲涅尔反射
                        ro = p + out_normal * 1e-4
                        rd = normalize(reflect(rd, out_normal))
                        # 纯玻璃反射不怎么损失能量
                        throughput *= obj_color 
                    else:
                        # 折射穿透
                        ro = p - out_normal * 1e-4  # ⚠️ 向内偏移防止自相交
                        rd = normalize(refract(rd, out_normal, etai_over_etat))
                        throughput *= obj_color

                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)
                    shadow_ray_orig = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
                    
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0
                        
                    ambient = 0.15 * obj_color
                    direct_light = ambient 
                    
                    if in_shadow == 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse
                    
                    final_color += throughput * direct_light
                    break
            
            pixel_color_acc += final_color
            
        # 平均多次采样的颜色，并写入像素
        pixels[i, j] = ti.math.clamp(pixel_color_acc / float(samples), 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Pro Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 2.0
    max_bounces[None] = 4       # 玻璃材质建议至少弹射4次，因为进出球体需要2次弹射
    samples_per_pixel[None] = 4 # 默认开启 4x MSAA 抗锯齿

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.72, 0.05, 0.26, 0.28):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
            samples_per_pixel[None] = gui.slider_int('AA Samples', samples_per_pixel[None], 1, 16)

        window.show()

if __name__ == '__main__':
    main()