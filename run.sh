stylegan2_pytorch --data ./data/ image_size 256 --aug-prob 0.25 --num_train_steps 180000 --batch-size 15 --gradient-accumulate-every 2 --name threedataset --log
stylegan2_pytorch --data ../../data/CombinedScene/ --batch-size 32 --gradient-accumulate-every 1 --fmap_max 256 --name threedatasetw0.1
stylegan2_pytorch --data ../../data/CombinedScene/ --batch-size 32 --gradient-accumulate-every 1 --fmap_max 256 --name twodatasetw1 --aug_prob 0.25 --network_capacity 8
stylegan2_pytorch --data ../../data/CombinedScene/ --batch-size 32 --gradient-accumulate-every 1 --name singledataset-big --aug_prob 0.25
stylegan2_pytorch --data ../../data/CombinedFace/ --batch-size 16 --gradient-accumulate-every 2 --name face256
stylegan2_pytorch --data ../../data/CombinedFace/ --batch-size 16 --gradient-accumulate-every 2 --name face256random
stylegan2_pytorch --data ../../data/CombinedFace/ --batch-size 16 --gradient-accumulate-every 2 --name face256age
stylegan2_pytorch --data ../../data/CombinedScene/ --batch-size 16 --gradient-accumulate-every 2 --name scene256-5attrnew-random-w5
stylegan2_pytorch --data ../../data/CombinedScene/ --batch-size 16 --gradient-accumulate-every 2 --name scene256-3attr-random-w3