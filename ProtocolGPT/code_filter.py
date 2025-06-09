import os
import re
import shutil
from utils import get_all_files
from consts import ALLOW_FILES

patterns = {
    "ikev2": r'ike_sa_state|exchange_type',
    "tls": r'message_type|state_machine',
    "bgp": r'session_state|session_events|msg_type',
    "rtsp": r'RTSP_Server_State|cur_state',
    "l2tp": r'state_machine|L2TP_LAIC_STATE|fsm_table',
}


#在协议实现中找到与状态机相关的文件
def get_fsm_files(dir, protocol):
    pattern = patterns[protocol]
    files = get_all_files(dir)
    fsm_files = []
    
    for filename in files:
        for ext in ALLOW_FILES:
            if filename.endswith(ext):
                with open(filename, 'r') as file:
                    text = file.read()
                # 在文件内容中查找匹配的内容
                matches = re.findall(pattern, text)
                if matches:
                    fsm_files.append(filename)
                    # print(filename)
    return fsm_files

#过滤掉与状态机无关的目录
def code_filter(dir, protocol):
    print("Filterring code in: " + dir)
    imple_name = dir.split("/")[-1]
    ignore_dirname = ["src"]
    fsm_subdirs = set()
    fsm_files = get_fsm_files(dir, protocol)
    for fsm_file in fsm_files:
        #去掉相对路径
        imple_file = fsm_file.replace(dir, "")
        sub_dirs = imple_file.split("/")[1:]
        
        #忽略测试代码
        if "test" in imple_file:
            continue 
        
        #如果二级目录是src，则保存三级目录
        if sub_dirs[0] in ignore_dirname:
            fsm_subdirs.add(sub_dirs[0] + "/" + sub_dirs[1])
        else:
            fsm_subdirs.add(sub_dirs[0])
    
    
    fsm_dirs = {dir + "/" + s for s in fsm_subdirs}
    print("FSM related files: ")
    os.makedirs(dir + "_filtered")
    try:
        for fsm_dir in fsm_dirs:
            if os.path.isdir(fsm_dir):
                shutil.copytree(fsm_dir, dir + "_filtered" + fsm_dir.replace(dir, ""))
            elif os.path.isfile(fsm_dir):
                shutil.copyfile(fsm_dir, dir + "_filtered" + fsm_dir.replace(dir, ""))
            print(fsm_dir)
    except Exception as e:
        print(e)

    
if __name__=="__main__":
    # code_filter("/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT/test_code/strongswan","ikev2")
    # code_filter("/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT/test_code/s2n-tls", "tls")
    # code_filter("/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT/test_code/openbgpd-openbsd", "bgp")
    code_filter("/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT/test_code/openbgpd-portable", "bgp")
    # code_filter("/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT/test_code/feng", "rtsp")
    # code_filter("/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT/test_code/openl2tp", "l2tp")
    

    