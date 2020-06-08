#!/bin/bash


### done
#models="
#unet_model1_elu
#unet_model1_relu
#unet_model_b5_elu
#unet_model_b5_relu

#unet_model1_elu_w10
#unet_model1_elu_w25
#unet_model1_relu_w10
#unet_model1_relu_w10_test
#unet_model1_relu_w25
#"

### done
#models="
#model_simple2_w10
#model_simple2_w25
#"

#models="
#unet_model1_elu_w10_test
#unet_model1_elu_w25_test
#unet_model1_relu_w10_test
#unet_model1_relu_w25_test
#"


### done - rerun completed
#models="
#resnet_model_elu_w10
#resnet_model_elu_w25
#resnet_model_relu_w10
#resnet_model_relu_w25
#"


### done - rerun completed
#models="
#resnet_model_s3_elu_w10
#resnet_model_s3_elu_w25
#resnet_model_s3_relu_w10
#resnet_model_s3_relu_w25
#"

### done
#models="
#unet_model_b5_elu_w10
#unet_model_b5_elu_w25
#unet_model_b5_relu_w10
#unet_model_b5_relu_w25
#"


#models="
#unet_model_b5_elu_bn_w10
#unet_model_b5_elu_bn_w25
#unet_model_b5_relu_bn_w10
#unet_model_b5_relu_bn_w25
#"

### 
#models="
#resnet_model_b5_elu_w10
#resnet_model_b5_elu_w25
#resnet_model_b5_relu_w10
#resnet_model_b5_relu_w25
#"

### 
#models="
#resnet_model_s3_b5_elu_w10
#resnet_model_s3_b5_elu_w25
#resnet_model_s3_b5_relu_w10
#resnet_model_s3_b5_relu_w25
#"


### 
#models="
#resnet_model_b5_relu_bn_w10
#resnet_model_b5_relu_bn_w25
#resnet_model_b5_elu_bn_w10
#resnet_model_b5_elu_bn_w25
#"

#models="
#unet_model1_elu_dice
#unet_model1_relu_dice
#"

### 
#models="
#inc_model_elu_w10
#inc_model_elu_w25
#inc_model_relu_w10
#inc_model_relu_w25
#"

#models="
#inc_model_b5_elu_w10
#inc_model_b5_elu_w25
#inc_model_b5_relu_w10
#inc_model_b5_relu_w25
#"


### done
#models="
#unet_model_b5_relu_do_w10
#unet_model_b5_relu_do_w25
#resnet_model_b5_relu_do_w10
#resnet_model_b5_relu_do_w25
#"

### done
#models="
#unet_model_b5_relu_us_w10
#unet_model_b5_relu_us_w25
#unet_model_b5_elu_us_w10
#unet_model_b5_elu_us_w25
#"

#models="
#resnet_model_b5_relu_w10_n750
#resnet_model_b5_relu_w10_n1000
#unet_model_b5_relu_w10_n750
#unet_model_b5_relu_w10_n1000
#"

#models="
#unet_model_b5_relu_p2p_w10
#unet_model_b5_relu_p2p_w25
#"

#models="
#resnet_model_b5_relu_dil_w10
#resnet_model_b5_relu_dil_w25
#"

#models="
#unet_model_b5_relu_dil_w10
#unet_model_b5_relu_dil_w25
#"

models="
unet_model_b5_relu_w10_n1500
resnet_model_b5_relu_w10_n1500
"


for model in $models
do
    echo $model
    python3 gen_mask.py --model_tag $model --data_tag "train" --n 500
    python3 gen_mask.py --model_tag $model --data_tag "test" --n 500
done


