import time
import threading
import numpy as np
import queue
import csv
import concurrent.futures
import os

JOB_NUM = 99  # 发送请求的个数

# 在opt-1.3B上的实验数据 单位: ms
x = [1, 4, 16, 64, 256, 512, 1024]
first_time = [5.88, 5.93, 6.57, 8.04, 23.8, 43.9, 98.5]
next_time = [5.13, 5.11, 5.16, 5.22, 5.52, 5.72, 5.82]

# 通过实验数据拟合每次迭代推理时间
z1 = np.polyfit(x, first_time, 1)
p1 = np.poly1d(z1)

z2 = np.polyfit(x, next_time, 1)
p2 = np.poly1d(z2)

#######################################################
# 创建全局队列，作为缓冲区使用
request_queue = queue.Queue()
# 和之前机器学习的最小二乘法使用方法一样，计算迭代时间
def fit_first_iter_time(prompt_length):
    return p1(prompt_length)

def fit_next_iter_time(prompt_length):
    return p2(prompt_length)



#######################################################

class Request:  # 推理请求，理论上输出长度未知，但为仿真实验，需要事先确定
    def __init__(self, j_id, prompt_length, output_length):
        self.j_id = j_id
        self.prompt_length = int(prompt_length)
        self.output_length = int(output_length)
        self.first_iter_time = fit_first_iter_time(self.prompt_length)
        self.next_iter_time  = fit_next_iter_time(self.output_length)
        self.iter_count = 0 # 请求执行了几次迭代，iter_count==output_length时完成整个推理   
        self.priority = -1  # 请求目前处于第几级队列
        
        self.create_time = time.time()  # 请求创建时间



class RequestGenerator(threading.Thread):

    def __init__(self, arrival_rate):
        super().__init__()
        self.arrival_rate = arrival_rate  # arrival rate = 1s / job interval
        
    def run(self):
        prompt_length_list = []
        output_length_list = []
        
        # 此处为读取orca数据集中的数据来构造request，可自行修改路径
        homedir = os.getcwd()
        f = open(homedir+'/orca_100k.csv', 'r')
        with f:
            reader = csv.reader(f)
            count=0
            for row in reader:
                if count == 0:
                    count += 1
                    continue
                try:
                    prompt_length_list.append(row[0])
                    output_length_list.append(row[1])
                except IndexError:
                    print(f"IndexError: 行 {row} 中的元素索引越界")
                except Exception as e:
                    print(f"发生了一个错误: {e}，行 {row}")
                # prompt_length_list.append(row[0])
                # output_length_list.append(row[1])
                
        j_id = 0
        
        while j_id < JOB_NUM:
            output_ = output_length_list[j_id]
            input_ = prompt_length_list[j_id]
            request = Request(j_id, input_, output_)
            request_queue.put(request)

            j_id += 1
            
            time.sleep(1 / self.arrival_rate)



# Define class

class SkipJoinMLFQScheduler:

    def __init__(self, first_quantum, quantum_rate, queue_num):
        # super().__init__()
        self.first_quantum=first_quantum
        self.quantum_rate=quantum_rate
        self.queue_num=queue_num
        self.quantum_list = []
        self.multi_level_priority_queue = []
        self.executed = 0  # 已经完成的请求数量
        # 论文中的quantum是在first_quantum上乘quantum_rate，和这里有点不一样
        # first quantum/Q1 is the min iteration time
        for i in range(self.queue_num):
            self.quantum_list.append(self.first_quantum*(self.quantum_rate ** i))
            temp_q = queue.Queue(-1) 
            self.multi_level_priority_queue.append(temp_q)
            
        self.ave_jct = []

    def getNewRequest(self, request: Request):
        # Todo: 处理缓冲区中新到达的request，根据他们的输入长度放入多级队列中
        for i in range(self.queue_num):
            if request.first_iter_time <=self.quantum_list[i] or i==self.queue_num-1:
                request.priority=i
                self.multi_level_priority_queue[i].put(request)
                # print(self.multi_level_priority_queue[i].qsize() )
                break

    
    def demoteRequest(self, job: Request):
        # Todo: 将完成了推理但还没生成完毕的请求放入下一级队列
        if job.iter_count < job.output_length:
            if job.priority < self.queue_num-1:
                job.priority+=1
            self.multi_level_priority_queue[job.priority].put(job)

    
    def getInferenceJob(self):
        # Todo: 返回在最高优先级的队列中的队首请求
        for q in self.multi_level_priority_queue:
            if not q.empty():
                return q.get()
        


def simulate_forward(iteration_time, job:Request, scheduler):
    
    iteration_num = scheduler.quantum_list[job.priority]  # 获取当前任务在这次推理中需要执行多少轮
    if job.iter_count==0:
        time.sleep(iteration_time / 1000)  # ms
        print("j_id:%d\ttoken:%d  \ttotal:%d"%(job.j_id,job.iter_count,job.output_length))
        job.iter_count += 1
        iteration_num -= iteration_time
        iteration_time = job.next_iter_time

    iteration_num = int(iteration_num//job.next_iter_time)
    if iteration_num >= job.output_length - job.iter_count:
        iteration_num = job.output_length - job.iter_count

        for i in range(iteration_num):
            time.sleep(iteration_time / 1000)  # ms
            print("j_id:%d\ttoken:%d  \ttotal:%d"%(job.j_id,job.iter_count,job.output_length))
            job.iter_count += 1

        jct = time.time() - job.create_time                     
        scheduler.ave_jct.append((job.j_id,jct))
        
        scheduler.executed += 1
        
    else:
        for i in range(iteration_num):
            time.sleep(iteration_time / 1000)  # ms
            print("j_id:%d\ttoken:%d  \ttotal:%d"%(job.j_id,job.iter_count,job.output_length))
            job.iter_count += 1

        scheduler.demoteRequest(job)



# 推理线程
def run(scheduler):
    while scheduler.executed != JOB_NUM:
        for i in range(request_queue.qsize()):
            req = request_queue.get()
            scheduler.getNewRequest(req)
        '''for i in range(4):
            if scheduler.multi_level_priority_queue[i].qsize()!=0:
                print(scheduler.multi_level_priority_queue[i].qsize())
        '''
        job = scheduler.getInferenceJob()
        if job==None:
            continue
        if job.iter_count == 0:
            iter_time = job.first_iter_time
        else:
            iter_time = job.next_iter_time

        args = [iter_time, job, scheduler]
        # 调用模拟推理线程
        temp_thread = thread_pool.submit(lambda p: simulate_forward(*p), args)


if __name__ == '__main__':
    arrival_rate=4
    quantum=6
    quantum_rate=2
    queue_num=8
    # 简化成一个时间一个推理任务，直接调用函数也可，这里为了方便修改max_workers，继续使用原代码
    thread_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    # 定义并启动发送请求的用户线程
    generator = RequestGenerator(arrival_rate=arrival_rate)
    generator.start()
    
    # 定义并启动调度器线程
    scheduler = SkipJoinMLFQScheduler(first_quantum=quantum,
                                      quantum_rate=quantum_rate,
                                      queue_num=queue_num)
    run(scheduler)
    
    for i in range(len(scheduler.ave_jct)):
        print(scheduler.ave_jct[i])
    values = np.array([x[1] for x in scheduler.ave_jct])
    print(np.mean(values))
    with open('example.txt', 'w') as f:
        for i in scheduler.ave_jct:
            f.write(str(i)+"\n")    
    thread_pool.shutdown()
    '''while 1:
        if len(scheduler.ave_jct)>=90:
            for i in scheduler.ave_jct:
                print(scheduler.ave_jct[i])
                '''

