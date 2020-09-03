from gym.envs.registration import register

# register(
#     id='mouse-v0',
#     entry_point='gym_mouse.envs:MouseEnv'
# )
register(
    id='mouseCl-v2',
    entry_point='gym_mouse.envs:MouseEnv_cl'
)

# Abort voxel
# register(
#     id='mousevox-v0',
#     entry_point='gym_mouse.envs:MouseEnv_vox'
# )