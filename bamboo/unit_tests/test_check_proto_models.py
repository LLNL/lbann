import sys, os, commands, re, pytest

def test_models():
    lbann_dir = commands.getstatusoutput('git rev-parse --show-toplevel')
    host = commands.getstatusoutput("hostname")
    host = re.sub("\d+", "", host[1])
    exe = lbann_dir[1] + '/build/' + host + '.llnl.gov/model_zoo/lbann'
    opt = lbann_dir[1] + '/model_zoo/optimizers/opt_adagrad.prototext'
    defective_models = []
    for subdir, dirs, files in os.walk(lbann_dir[1] + '/model_zoo/models/'):
        for file_name in files:
            if file_name.endswith('.prototext') and "model" in file_name:
                model_path = subdir + '/' + file_name
                print('Attempting model setup for: ' + file_name )
                if 'mnist' in file_name:
                    cmd = exe + ' --model=' + model_path + ' --reader='+ lbann_dir[1] + '/model_zoo/data_readers/data_reader_mnist.prototext' + ' --optimizer=' + opt + ' --exit_after_setup'
                    if os.system(cmd) != 0:
                        print("Error detected in " + model_path)
                        defective_models.append(file_name)
                elif 'net' in file_name:
                    cmd = exe + ' --model=' + model_path + ' --reader='+ lbann_dir[1] + '/model_zoo/data_readers/data_reader_imagenet.prototext' + ' --optimizer=' + opt + ' --exit_after_setup'
                    if os.system(cmd) != 0:
                        print("Error detected in " + model_path)
                        defective_models.append(file_name)       
                elif 'cifar' in file_name:
                    cmd = exe + ' --model=' + model_path + ' --reader='+ lbann_dir[1] + '/model_zoo/data_readers/data_reader_cifar10.prototext' + ' --optimizer=' + opt + ' --exit_after_setup'
                    if os.system(cmd) != 0:
                        print("Error detected in " + model_path)
                        defective_models.append(file_name)
                else:
                    #cmd = exe + ' --model=' + model_path + ' --exit_after_setup'
                    #if os.system(cmd) != 0:
                    #print("Error detected in " + model_path)                
                    print("Tell Dylan which data reader this model needs")
    if len(defective_models) != 0:
        print("The following models exited with errors")
        for i in defective_models:
            print(i)
    assert len(defective_models) == 0
