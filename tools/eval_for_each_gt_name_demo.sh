CFG_FILE="/nas2/YJ/git/OpenPCDet/output/tree_models/1st/pointpillar_base/default/pointpillar_base.yaml"
CKPT="/nas2/YJ/git/OpenPCDet/output/tree_models/1st/pointpillar_base/default/ckpt/checkpoint_epoch_200.pth"
python eval_for_each_gt_name_demo.py \
        --cfg_file ${CFG_FILE} \
        --ckpt ${CKPT} \
        --txt_file ./../data/tree/ImageSets/test.txt \
        --path testing \
        --name filename_0.3  \
	--iou 0.3

