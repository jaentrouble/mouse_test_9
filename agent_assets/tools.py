import cv2
from os import path
import numpy as np

def evaluate_mouse(player, env, video_type):
    print('Evaluating...')
    done = False
    video_dir = path.join(player.model_dir, 'eval.{}'.format(video_type))
    eye_dir = path.join(player.model_dir, 'eval_eye.{}'.format(video_type))
    score_dir = path.join(player.model_dir, 'score.txt')
    if 'avi' in video_type :
        fcc = 'DIVX'
    elif 'mp4' in video_type:
        fcc = 'mp4v'
    else:
        raise TypeError('Wrong videotype')
    fourcc = cv2.VideoWriter_fourcc(*fcc)
    # Becareful : cv2 order of image size is (width, height)
    eye_out = cv2.VideoWriter(eye_dir, fourcc, 10, (205*5,50))
    out = cv2.VideoWriter(video_dir, fourcc, 10, env.image_size)
    eye_bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)
    o = env.reset()
    score = 0
    loop = 0
    while not done :
        loop += 1
        if not loop % 100:
            print('Eval : {}step passed'.format(loop))
        a = player.act(o, record=False)
        o,r,done,i = env.step(a)
        score += r
        #eye recording
        rt_eye = np.flip(o['Right'][:,-1,:],axis=0)
        lt_eye = o['Left'][:,-1,:]
        eye_img = np.concatenate((lt_eye,eye_bar,rt_eye))
        eye_img = np.broadcast_to(eye_img.reshape((1,205,1,3)),(50,205,5,3))
        eye_img = eye_img.reshape(50,205*5,3)
        eye_out.write(np.flip(eye_img, axis=-1))
        # This will turn image 90 degrees, but it does not make any difference,
        # so keep it this way to save computations
        out.write(np.flip(env.render('rgb'), axis=-1))
    out.release()
    eye_out.release()
    with open(score_dir, 'w') as f:
        f.write(str(score))
    print('Eval finished')
    return score

def evaluate_common(player, env, video_type):
    print('Evaluating...')
    done = False
    video_dir = path.join(player.model_dir, 'eval.{}'.format(video_type))
    score_dir = path.join(player.model_dir, 'score.txt')
    if 'avi' in video_type :
        fcc = 'DIVX'
    elif 'mp4' in video_type:
        fcc = 'mp4v'
    else:
        raise TypeError('Wrong videotype')
    fourcc = cv2.VideoWriter_fourcc(*fcc)
    # Becareful : cv2 order of image size is (width, height)
    o = env.reset()
    rend_img = env.render('rgb_array')
    # cv2 expects 90 degrees rotated
    out_shape = (rend_img.shape[1],rend_img.shape[0])
    out = cv2.VideoWriter(video_dir, fourcc, 10, out_shape)
    score = 0
    loop = 0
    while not done :
        loop += 1
        if loop % 100 == 0:
            print('Eval : {}step passed'.format(loop))
        a = player.act(o, record=False)
        o,r,done,i = env.step(a)
        score += r
        # This will turn image 90 degrees, but it does not make any difference,
        # so keep it this way to save computations
        img = env.render('rgb_array')
        out.write(np.flip(env.render('rgb_array'), axis=-1))
    out.release()
    with open(score_dir, 'w') as f:
        f.write(str(score))
    print('Eval finished')
    env.close()
    return score
