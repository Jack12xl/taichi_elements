import taichi as ti
import os
import sys
import time
from pathlib import Path
sys.path.append(sys.path[0]+'/renderer')
from engine.renderer import res, Renderer

ti.init(arch=ti.cuda, use_unified_memory=False, device_memory_fraction=0.8)

renderer = Renderer(dx=1 / 512, shutter_time=2e-3, taichi_logo=False)
spp=100

input_folder = sys.argv[1]
output_folder = sys.argv[2]
for f in range(26):
    input_fn = f'{input_folder}/test_frame{f:03d}.bin'
    print('input from', input_fn)
    output_fn = f'{output_folder}/{f:05d}.png'
    print('frame', f, end='')
    if os.path.exists(output_fn):
        print('skip.')
        continue
    else:
        print('rendering...')
    Path(output_fn).touch()
    t = time.time()
    renderer.initialize_particles_from_bin(input_fn)
    img = renderer.render_frame(spp=spp)
    ti.imwrite(img, output_fn)
    print(f'Frame rendered. {spp} take {time.time() - t} s.')

