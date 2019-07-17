""" Racsac 伪代码
def ransacMatching(A, B):
    A & B: List of List
    A 样本集 对应 B为结果集
    consensus_set 内点集
    n 计算模型需要的最小数据个数,sift特征点匹配中n=4
    model 通过n计算出来的模型
    k 迭代次数
    error model计算结果和B的误差
    t 决定数据是否适用model的阀点值
    d 判断model是否OK的内点集数据
    best_model 满足d的最佳模型

    consensus_set=[]
    i=0
    while i < k:
        a=np.random.choice(A,n,True) #在data中随机抽取n个数据
        model=a（...） 通过随机样本a计算model
        temp_set=[] 临时内点集为空集
        for i in A: #i为n个数
            error= model(i) -B  将i带入model并进行误差计算
            if error < t:
                temp_set.append([i])  如果误差小于判定阈值，则i属于内点集
        if len[temp_set]> len[consensus_set]
            consensus_set=temp_set 迭代k次之后，选择出最大的内点集

    在新的内点集中重新迭代model
    i=0
    error=无穷大  初始误差为0
    while i < k: #i为n个数,即计算模最数
        a=np.random.choice(consensus_set,n,True) #在内点集中随机抽取n个数据
        model=a（...） 通过随机样本a计算model
        temp_error=0 初始误差为0
        for i in consensus_set:
            temp_error+= model(i) -B  将i带入model并进行误差计算
        if temp_error<error:
            error=temp_error 将误差小的值替换初始error
            best_model=model 将对应的model替换成best_model
    return (best_model, error, consensus_set)  输出最佳模型，对应错误率，对应最大的内点集

"""