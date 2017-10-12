import os, sys, re, commands 
import pytest


def run_lbann(executable, test='accuracy_test.sh',model='mnist',optimizer='adagrad',epochs=2,nodes=1,procs=2,iterative=0):
    lbann_dir = commands.getstatusoutput('git rev-parse --show-toplevel')
    test_cmd = lbann_dir[1] + '/bamboo/integration_tests/accuracy_tests/' + test
    CMD = test_cmd + ' -exe ' + executable +  ' -m ' + model + ' -o ' + optimizer + ' -N %d -n %d -e %d'% (nodes, procs, epochs)
    if iterative == 1:
        CMD = CMD + ' -i'
        print CMD
    if os.system(CMD) != 0:
        #print(CMD)
        print 'LBANN failed during execution. Please check the Bamboo build log'
        sys.exit(1)
        


def data_format(res_file):
    with open(res_file) as f:
        acc = map(float,f)
    return acc

def fetch_true(master,model,model_num,epochs):
    true_acc=[]
    header = model.upper() + ' ' + model_num + ' ' + epochs 
    with open(master) as f:
        for line in f:
            if header in line:
                for i in range(int(model_num)):
                    true_acc.append(next(f).strip('\n'))
    true_acc = [float(i) for i in true_acc]
    return true_acc


def test_accuracy_mnist(log,exe):
    
    # use run_lbann calls here to generate accuracy files. All generated accuracies will be tested in the below assertion loop
    # Default to MNIST with adagrad and 5 epochs. 
    run_lbann(exe)
    general_assert('mnist') 
    
    #if log == 0:
    #    os.system("rm res_mnist*")

def test_accuracy_alexnet(log,exe):
    run_lbann(exe, model='alexnet')
    general_assert('alexnet')
    if log == 0:
        os.system("rm res_alexnet*")
#    run_lbann(iterative=1,model='resnet')
#    general_assert('resnet')
#    if log == 0:
#        os.system("rm trimmed*")

def general_assert(model):
    for filename in os.listdir(os.getcwd()):
        if filename.startswith('res_'+model):
            #intro = re.search('trimmed_res_',filename)
            #trailing = re.search('[0-9]',filename)
            #model = filename[intro.end():trailing.start()-1]
            numbers = re.findall('[0-9]',filename)
            model_num = numbers[0]
            epochs = numbers[1]

            test_acc = data_format(filename)
            true_acc = fetch_true('integration_tests/accuracy_tests/masters.txt',model,model_num,epochs)
            for test, true in zip(test_acc,true_acc):
                assert true <= test
