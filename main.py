# %%
from kmeans import*

def main(argv):
    if len(argv)<3:
        print("please type $python main.py, CSVfile, clusterNum, NormNum")
        return
    #k: setting group numbers
    k=int(argv[2])
    if k<=0:
        print("group/cluster number must be >=1")
        return
    if not Utils.CheckLabelNums(k):
        print("please check label numbers if they are enough")
        return

    model = KMeans(argv[1],k)
    if model.utils.init:
        #2:L2, 1:L1 norm
        norm = int(argv[3])
        model.Fit(norm)
        model.Evaluate(norm)
    else:
        print("init error")

if __name__=="__main__":
    main(sys.argv[0:])

        