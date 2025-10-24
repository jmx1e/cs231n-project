# all commands to run for all results in the paper
# forward operators: hdr (symmetric), hdr_highlight_clip (overexposed), hdr_shadow_clip (underexposed)
# sampling methods/models: pixel, ldm, sd
# data: ffhq, imagenet
# metrics: psnr, ssim, lpips

# all sampling was done on Google Compute Engine
# Machine type: g2-standard-8 (8 vCPUs, 32 GB Memory)
# GPU: 1 x NVIDIA L4 with 24 GB of memory

# approximate sampling times are listed below each command in hh:mm:ss format


# 00:30:10 symmetric, ffhq, pixel-4k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/symmetric num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-ffhq gpu=0

# 00:07:30 symmetric, ffhq, pixel-1k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/symmetric num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=pixel1-ffhq gpu=0

# 01:28:05 symmetric, ffhq, ldm-1k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ldm +task=hdr +sampler=latent_edm_daps task_group=ldm save_dir=results/symmetric num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=ldm-ffhq gpu=0

# 03:23:05 symmetric, ffhq, sd1.5-1k
python posterior_sample.py +data=demo-ffhq +model=stable-diffusion-v1.5 +task=hdr +sampler=sd_edm_daps task_group=sd save_dir=results/symmetric num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=sd15-ffhq gpu=0

# 01:53:35 symmetric, imagenet, pixel-4k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/symmetric num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-imagenet gpu=0

# 00:28:20 symmetric, imagenet, pixel-1k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/symmetric num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=pixel1-imagenet gpu=0

# 01:28:20 symmetric, imagenet, ldm-1k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ldm +task=hdr +sampler=latent_edm_daps task_group=ldm save_dir=results/symmetric num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=ldm-imagenet gpu=0

# 03:22:45 symmetric, imagenet, sd1.5-1k
python posterior_sample.py +data=demo-imagenet +model=stable-diffusion-v1.5 +task=hdr +sampler=sd_edm_daps task_group=sd save_dir=results/symmetric num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=sd15-imagenet gpu=0


# 00:30:30 overexposed, ffhq, pixel-4k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr_highlight_clip +sampler=edm_daps task_group=pixel save_dir=results/over num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-ffhq gpu=0

# 00:07:35 overexposed, ffhq, pixel-1k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr_highlight_clip +sampler=edm_daps task_group=pixel save_dir=results/over num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=pixel1-ffhq gpu=0

# 01:28:05 overexposed, ffhq, ldm-1k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ldm +task=hdr_highlight_clip +sampler=latent_edm_daps task_group=ldm save_dir=results/over num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=ldm-ffhq gpu=0

# 03:23:45 overexposed, ffhq, sd1.5-1k
python posterior_sample.py +data=demo-ffhq +model=stable-diffusion-v1.5 +task=hdr_highlight_clip +sampler=sd_edm_daps task_group=sd save_dir=results/over num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=sd15-ffhq gpu=0

# 01:54:05 overexposed, imagenet, pixel-4k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ddpm +task=hdr_highlight_clip +sampler=edm_daps task_group=pixel save_dir=results/over num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-imagenet gpu=0

# 00:28:30 overexposed, imagenet, pixel-1k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ddpm +task=hdr_highlight_clip +sampler=edm_daps task_group=pixel save_dir=results/over num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=pixel1-imagenet gpu=0

# 01:28:20 overexposed, imagenet, ldm-1k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ldm +task=hdr_highlight_clip +sampler=latent_edm_daps task_group=ldm save_dir=results/over num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=ldm-imagenet gpu=0

# 03:22:20 overexposed, imagenet, sd1.5-1k
python posterior_sample.py +data=demo-imagenet +model=stable-diffusion-v1.5 +task=hdr_highlight_clip +sampler=sd_edm_daps task_group=sd save_dir=results/over num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=sd15-imagenet gpu=0


# 00:30:35 underexposed, ffhq, pixel-4k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr_shadow_clip +sampler=edm_daps task_group=pixel save_dir=results/under num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-ffhq gpu=0

# 00:07:40 underexposed, ffhq, pixel-1k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr_shadow_clip +sampler=edm_daps task_group=pixel save_dir=results/under num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=pixel1-ffhq gpu=0

# 01:28:05 underexposed, ffhq, ldm-1k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ldm +task=hdr_shadow_clip +sampler=latent_edm_daps task_group=ldm save_dir=results/under num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=ldm-ffhq gpu=0

# 03:23:00 underexposed, ffhq, sd1.5-1k
python posterior_sample.py +data=demo-ffhq +model=stable-diffusion-v1.5 +task=hdr_shadow_clip +sampler=sd_edm_daps task_group=sd save_dir=results/under num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=sd15-ffhq gpu=0

# 01:54:05 underexposed, imagenet, pixel-4k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ddpm +task=hdr_shadow_clip +sampler=edm_daps task_group=pixel save_dir=results/under num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-imagenet gpu=0

# 00:28:25 underexposed, imagenet, pixel-1k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ddpm +task=hdr_shadow_clip +sampler=edm_daps task_group=pixel save_dir=results/under num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=pixel1-imagenet gpu=0

# 01:28:25 underexposed, imagenet, ldm-1k
python posterior_sample.py +data=demo-imagenet +model=imagenet256ldm +task=hdr_shadow_clip +sampler=latent_edm_daps task_group=ldm save_dir=results/under num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=ldm-imagenet gpu=0

# 03:22:00 underexposed, imagenet, sd1.5-1k
python posterior_sample.py +data=demo-imagenet +model=stable-diffusion-v1.5 +task=hdr_shadow_clip +sampler=sd_edm_daps task_group=sd save_dir=results/under num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=sd15-imagenet gpu=0





# ablation study for NFE with pixel on ffhq: 50, 100, 200, 400, 1k, 2k, 4k

# 00:00:30 for pixel-50
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/ablation num_runs=1 sampler.diffusion_scheduler_config.num_steps=2 sampler.annealing_scheduler_config.num_steps=25 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-50 gpu=0

# 00:01:05 for pixel-100
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/ablation num_runs=1 sampler.diffusion_scheduler_config.num_steps=2 sampler.annealing_scheduler_config.num_steps=50 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-100 gpu=0

# 00:02:10 for pixel-200
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/ablation num_runs=1 sampler.diffusion_scheduler_config.num_steps=2 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-200 gpu=0

# 00:03:30 for pixel-400
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/ablation num_runs=1 sampler.diffusion_scheduler_config.num_steps=4 sampler.annealing_scheduler_config.num_steps=100 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-400 gpu=0

# 00:08:20 for pixel-1k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/ablation num_runs=1 sampler.diffusion_scheduler_config.num_steps=5 sampler.annealing_scheduler_config.num_steps=200 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-1k gpu=0

# 00:15:30 for pixel-2k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/ablation num_runs=1 sampler.diffusion_scheduler_config.num_steps=8 sampler.annealing_scheduler_config.num_steps=250 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-2k gpu=0

# 00:30:10 for pixel-4k
python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/ablation num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=2 data.start_id=0 data.end_id=10 name=pixel-4k gpu=0
