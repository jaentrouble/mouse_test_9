__kernel void ray(
    __global const uchar *image,
    int img_x, int img_y, int ray_num, int light_ratio,
    __global const uchar *bg_col, __global const uchar *wall_col,
    __global const float *fp_ray, __global const float *delta_vec,
    __global uchar *observation
)
{
    int eye = get_global_id(0);
    int ray_idx = get_global_id(1);
    float fp_x = fp_ray[eye*2*ray_num + ray_idx];
    float fp_y = fp_ray[eye*2*ray_num + ray_num + ray_idx];
    float del_x = delta_vec[eye*2*ray_num + ray_idx];
    float del_y = delta_vec[eye*2*ray_num + ray_num + ray_idx];

    float2 fp_pos = (float2) (fp_x, fp_y);
    float2 delta = (float2) (del_x, del_y);

    int2 newpos;
    int conv_pos;
    uchar3 my_color = (uchar3) (wall_col[0], wall_col[1], wall_col[2]);

    bool loop = true;
    int dist = 0;

    while (loop) {
        dist ++;
        fp_pos += delta;
        if (fp_pos.x >= (img_x - 1)) {
            fp_pos.x = img_x - 1;
            loop = false;
        } else if (fp_pos.x <= 0) {
            fp_pos.x = 0;
            loop = false;
        }
        if (fp_pos.y >= (img_y - 1)) {
            fp_pos.y = img_y - 1;
            loop = false;
        } else if (fp_pos.y <= 0) {
            fp_pos.y = 0;
            loop = false;
        }

        newpos = convert_int2(round(fp_pos));
        conv_pos = newpos.x*img_y*3 + newpos.y*3;
        if (!((image[conv_pos+0] == bg_col[0]) &&
            (image[conv_pos+1] == bg_col[1]) &&
            (image[conv_pos+2] == bg_col[2]))) {
                my_color.x = image[conv_pos+0];
                my_color.y = image[conv_pos+1];
                my_color.z = image[conv_pos+2];
                loop = false;
            }
    }

    observation[eye*ray_num*3+ray_idx*3] = my_color.x/(dist/light_ratio +1);
    observation[eye*ray_num*3+ray_idx*3+1] = my_color.y/(dist/light_ratio +1);
    observation[eye*ray_num*3+ray_idx*3+2] = my_color.z/(dist/light_ratio +1);
}