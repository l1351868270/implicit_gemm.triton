import os
import pandas as pd

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build/best_conv2d')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cutlass_profiler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cutlass/build/tools/profiler/cutlass_profiler')
    input_args = [{'n': 1, 'h': 56, 'w': 56, 'c': 64, 'k': 64, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 56, 'w': 56, 'c': 64, 'k': 64, 'r': 3, 's': 3, 'pad_h': 1, 'pad_w': 1, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 56, 'w': 56, 'c': 64, 'k': 256, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 56, 'w': 56, 'c': 64, 'k': 256, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 56, 'w': 56, 'c': 256, 'k': 64, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 56, 'w': 56, 'c': 256, 'k': 128, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  
                  {'n': 1, 'h': 28, 'w': 28, 'c': 128, 'k': 128, 'r': 3, 's': 3, 'pad_h': 1, 'pad_w': 1, 'stride_h': 2, 'stride_w': 2, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 28, 'w': 28, 'c': 128, 'k': 512, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 28, 'w': 28, 'c': 256, 'k': 512, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 2, 'stride_w': 2, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 28, 'w': 28, 'c': 512, 'k': 128, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 28, 'w': 28, 'c': 128, 'k': 128, 'r': 3, 's': 3, 'pad_h': 1, 'pad_w': 1, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                  {'n': 1, 'h': 28, 'w': 28, 'c': 512, 'k': 256, 'r': 1, 's': 1, 'pad_h': 0, 'pad_w': 0, 'stride_h': 1, 'stride_w': 1, 'dilation_h': 1, 'dilation_w': 1},
                 ] 
    
    outputs = []
    outputs_names = []
    for cmd_args in input_args:
        outputs_name = f'{cmd_args["n"]}x{cmd_args["h"]}x{cmd_args["w"]}x{cmd_args["c"]}x{cmd_args["k"]}x{cmd_args["r"]}x{cmd_args["s"]}x{cmd_args["pad_h"]}x{cmd_args["pad_w"]}x{cmd_args["stride_h"]}x{cmd_args["stride_w"]}x{cmd_args["dilation_h"]}x{cmd_args["dilation_w"]}'
        outputs_names.append(outputs_name)
        output = f'{output_dir}/best_{outputs_name}'
        cmd = f'{cutlass_profiler_path} --operation=Conv2d --Activation=f16:nhwc --Filter=f16:nhwc --Output=f16 --accumulator-type=f32 --conv_kind=fprop ' \
              f'--output={output} --n={cmd_args["n"]} ' \
              f'--h={cmd_args["h"]} --w={cmd_args["w"]} --c={cmd_args["c"]} --k={cmd_args["k"]} ' \
              f'--r={cmd_args["r"]} --s={cmd_args["s"]} --pad_h={cmd_args["pad_h"]} --pad_w={cmd_args["pad_w"]} ' \
              f'--stride_h={cmd_args["stride_h"]} --stride_w={cmd_args["stride_w"]} --dilation_h={cmd_args["dilation_h"]} --dilation_w={cmd_args["dilation_w"]}'
        outputs.append(output)
        print(cmd)
        result = os.popen(cmd)
        result.read()

    best_conv2d = []
    for outputs_name, output in zip(outputs_names, outputs):
        df = pd.read_csv(f'{output}.conv2d.csv').sort_values(by='GFLOPs', ascending=False)
        best_0 = f"{outputs_name},{df.iloc[0]['Operation']},{df.iloc[0]['GFLOPs']}"
        best_1 = f"{outputs_name},{df.iloc[1]['Operation']},{df.iloc[1]['GFLOPs']}"
        best_2 = f"{outputs_name},{df.iloc[2]['Operation']},{df.iloc[2]['GFLOPs']}"
        best_conv2d.append(best_0)
        best_conv2d.append(best_1)
        best_conv2d.append(best_2)
        best_conv2d.append("+,+,+")
    
    with open(f'{output_dir}/best_conv2d.csv', 'w') as f:
        f.writelines('\n'.join(best_conv2d))