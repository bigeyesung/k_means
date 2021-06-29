# %%
from kmeans import*

def main(argv):
    if len(argv)<3:
        print("please type python main.py, CSVfile, clusterNum, NormNum")
        return
    if int(argv[2])<=0:
        print("k paramater must be >=1")
        return
    #k: setting group numbers
    k=int(argv[2])
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

        