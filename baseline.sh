# For HDR symmetric clipping baseline
python baseline_reconstruction.py +data=demo-ffhq +task=hdr task_group=pixel save_dir=results/baselines batch_size=2 data.start_id=0 data.end_id=10 name=hdr-baseline-ffhq gpu=0
python baseline_reconstruction.py +data=demo-imagenet +task=hdr task_group=pixel save_dir=results/baselines batch_size=2 data.start_id=0 data.end_id=10 name=hdr-baseline-imagenet gpu=0

# For HDR highlight clipping baseline  
python baseline_reconstruction.py +data=demo-ffhq +task=hdr_highlight_clip task_group=pixel save_dir=results/baselines batch_size=2 data.start_id=0 data.end_id=10 name=hdr-highlight-baseline-ffhq gpu=0
python baseline_reconstruction.py +data=demo-imagenet +task=hdr_highlight_clip task_group=pixel save_dir=results/baselines batch_size=2 data.start_id=0 data.end_id=10 name=hdr-highlight-baseline-imagenet gpu=0

# For HDR shadow clipping baseline
python baseline_reconstruction.py +data=demo-ffhq +task=hdr_shadow_clip task_group=pixel save_dir=results/baselines batch_size=2 data.start_id=0 data.end_id=10 name=hdr-shadow-baseline-ffhq gpu=0
python baseline_reconstruction.py +data=demo-imagenet +task=hdr_shadow_clip task_group=pixel save_dir=results/baselines batch_size=2 data.start_id=0 data.end_id=10 name=hdr-shadow-baseline-imagenet gpu=0