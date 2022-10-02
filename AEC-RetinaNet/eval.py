from object_detection_fastai.helper.wsi_loader import *
from my_RetinaNetFocalLossDA import my_RetinaNetFocalLossDAA
from my_RetinaNet import my_RetinaNetDA
from utils.domain_adaptation_helper import *
from custom_callbacks import UpdateAlphaCallback
from utils.slide_helper import load_images

import matplotlib.pyplot as pltx
from tqdm import tqdm

def get_y_func(x):
    return x.y

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    slide_folder = '/media/midog/Medal2/MIDOG_2021'
    model_dir = Path("/home/midog/Desktop/saketh/MIDOG-main/model_logs")

    patch_size = 512
    res_level = 0
    bs = 24
    domain_weight = 25
    lr = 1e-4
    train_samples_per_scanner = 10
    val_samples_per_scanner = 1000
    scales = [0.2, 0.4, 0.6, 0.8, 1.0]
    ratios = [1]
    sizes = [(64, 64), (32, 32), (16, 16)]
    num_epochs = 1

    train_scanners = [["A","B","D"]]
    valid_scanners = [["A","B","C"]]
    annotation_json ='/media/midog/Medal2/MIDOG_2021/MIDOG.json'

    scan_id={}
    scan_id['A']=0
    scan_id['B']=1
    scan_id['C']=2
    scan_id['D']=3

    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # tfms = get_transforms(do_flip=True,
    #                       flip_vert=True,
    #                       max_lighting=0.5,
    #                       max_zoom=2,
    #                       max_warp=0.2,
    #                       p_affine=0.5,
    #                       p_lighting=0.5,
    #                       )


    for t_scrs, v_scrs in zip (train_scanners, valid_scanners):
        learner_name = 'my_pre10_RetinaNet_ABD'
        train_images = []
        valid_images = []

        for t,ts in enumerate(t_scrs):
            train_container = load_images(Path("{}/{}/{}".format(slide_folder,ts,"train")), annotation_json, res_level, patch_size, scan_id[ts], categories = [1,2])
            train_samples = list(np.random.choice(train_container, train_samples_per_scanner))
            train_images.append(train_samples)
        for v,vs in enumerate(v_scrs):
            valid_container = load_images(Path("{}/{}/{}".format(slide_folder,vs,"val")), annotation_json, res_level, patch_size, scan_id[vs], categories = [1,2])
            valid_samples = list(np.random.choice(valid_container, val_samples_per_scanner))
            valid_images.append(valid_samples)
        train_images = [sub[item] for item in range(len(train_images[0]))for sub in train_images]
        valid_images = [sub[item] for item in range(len(valid_images[0])) for sub in valid_images]
        train = ObjectItemListSlide(train_images)
        valid = ObjectItemListSlide(valid_images)
        item_list = ItemLists(slide_folder, train, valid)
        lls = item_list.label_from_func(get_y_func, label_cls=SlideObjectCategoryListDA)  #
        # lls = lls.transform(tfms, tfm_y=True, size=patch_size)
        data = lls.databunch(bs=bs, collate_fn=bb_pad_collate_da, num_workers=0).normalize()
        data.train_dl = data.train_dl.new(shuffle=False) #set shuffle to false so that batch always contains all 4 scanners
        data.valid_dl = data.valid_dl.new(shuffle=False)
        anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
        #crit = my_RetinaNetFocalLossDAA(anchors, domain_weight=domain_weight, n_domains=len(t_scrs))
        encoder = create_body(models.resnet18, True, -2)
        # Careful: Number of anchors has to be adapted to scales
        model = my_RetinaNetDA(encoder, n_classes=data.train_ds.c, n_domains=len(t_scrs), n_anchors=len(scales) * len(ratios),
                            sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3, imsize = (patch_size, patch_size))
        voc = PascalVOCMetricByDistanceDAA(anchors, patch_size,[str(i) for i in data.train_ds.y.classes[1:]])

        #learn = Learner(data, model, loss_func=crit, metrics=[voc], callback_fns=[ShowGraph, BBBMetrics])
    #learn = load_learner(model_dir,'DA_RetinaNet.pkl')
    #print(learn.metrics)
    #learn.data.valid_dl = data.valid_dl

    # it =iter(data.valid_dl)
    # data=next(it)
    # data=next(it)
    # print(data[0].size())
    # for i in range(3):
    #     print(data[1][i])
    #exit()

    #print(learn.predict(data,with_input=True))
    #learn.names = ifnone(learn.loss_func.metric_names, [])
    #print(learn.names)
    # l=learn.get_preds()
    # print(len(l[0][0]))
    # print(len(l[1][0]))
    # for i in range(3):
    #     if i<=1:
    #         print(l[0][0][i].size())
    #     elif i==2:
    #         print(l[0][0][2])
    # print(l[0][0][3])
    # for i in range(4):
    #     print(l[1][0][i])
    # # print(l[1][0].size())
    # for i in range(5):
    #     print("Starting manual evaluation")
    #     with torch.no_grad():
    #         learn.model.eval()
    #         voc.on_epoch_begin()
    #         for x,y in tqdm(learn.data.valid_dl):
    #             #print(count)
    #             x=x.to(device)
    #             pred=learn.model(x)
    #             voc.on_batch_end(pred,y)
    #             #count+=1
    #         print(voc.man_end())


    # model=torch.load('/home/midog/Desktop/saketh/MIDOG-main/model_logs/models/'+learner_name+'_last.pth')
    s=torch.load('/home/midog/Desktop/saketh/MIDOG-main/model_logs/models/'+learner_name+'.pth')
    model.to(device)
    model.load_state_dict(s['model'])
    print('Loaded')

    for i in range(1):
        with torch.no_grad():
            model.eval()
            voc.on_epoch_begin()
            for x,y in tqdm(data.valid_dl):
                #print(count)
                x=x.to(device)
                pred=model(x)
                voc.on_batch_end(pred,y)
                #count+=1
            res=voc.on_man_end()
        print('F1_0: ',res[0]['F1'])
        print('AP_0: ',res[0]['AP'])
        print('F1_1: ',res[1]['F1'])
        print('AP_1: ',res[1]['AP'])
        torch.save(res,'/home/midog/Desktop/saketh/MIDOG-main/model_logs/evals/NTT_'+learner_name+'.pth.tar')
        # torch.save(res,'/home/midog/Desktop/saketh/MIDOG-main/model_logs/evals/NTT_'+learner_name+'_last.pth.tar')

