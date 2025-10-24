python posterior_sample.py +data=demo-ffhq +model=ffhq256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/simple num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=1 data.start_id=0 data.end_id=1 name=hdr-pixel-ffhq gpu=0

python posterior_sample.py +data=demo-ffhq +model=ffhq256ldm +task=hdr +sampler=latent_edm_daps task_group=ldm save_dir=results/simple num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=1 data.start_id=0 data.end_id=1 name=hdr-ldm-ffhq gpu=0

python posterior_sample.py +data=demo-ffhq +model=stable-diffusion-v1.5 +task=hdr +sampler=sd_edm_daps task_group=sd save_dir=results/simple num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=1 data.start_id=0 data.end_id=1 name=hdr-sd15-ffhq gpu=0

python posterior_sample.py +data=demo-imagenet +model=imagenet256ddpm +task=hdr +sampler=edm_daps task_group=pixel save_dir=results/simple num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=400 batch_size=1 data.start_id=0 data.end_id=1 name=hdr-pixel-imagenet gpu=0

python posterior_sample.py +data=demo-imagenet +model=imagenet256ldm +task=hdr +sampler=latent_edm_daps task_group=ldm save_dir=results/simple num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=1 data.start_id=0 data.end_id=1 name=hdr-ldm-imagenet gpu=0

python posterior_sample.py +data=demo-imagenet +model=stable-diffusion-v1.5 +task=hdr +sampler=sd_edm_daps task_group=sd save_dir=results/simple num_runs=1 sampler.diffusion_scheduler_config.num_steps=10 sampler.annealing_scheduler_config.num_steps=100 batch_size=1 data.start_id=0 data.end_id=1 name=hdr-sd15-imagenet gpu=0