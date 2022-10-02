from object_detection_fastai.callbacks.callbacks import *
from object_detection_fastai.helper.wsi_loader import SlideObjectItemList,SlideContainer


class DomainAdaptationItem(ItemBase):
    def __init__(self, imagebbox):
        self.imagebbox = imagebbox
        self.scanner_id = imagebbox.sample_kwargs["domain"]
        self.obj = (imagebbox, self.scanner_id)
        self.data = [imagebbox.data]

    def apply_tfms(self, tfms, **kwargs):
        self.imagebbox = self.imagebbox.apply_tfms(tfms, **kwargs)
        self.obj = (self.imagebbox, self.scanner_id)
        self.data = [self.imagebbox.data]
        return self

class SlideObjectCategoryListDA(ObjectCategoryList):

    def get(self, i, x: int = 0, y: int = 0):
        h, w = self.x.items[i].shape
        bboxes, labels = self.items[i]

        bboxes = np.array([box for box in bboxes]) if len(np.array(bboxes).shape) == 1 else np.array(bboxes)
        labels = np.array(labels)

        if len(labels) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y

            bb_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2
            bb_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2

            ids = ((bboxes[:, 0] + bb_widths) > 0) \
                  & ((bboxes[:, 1] + bb_heights) > 0) \
                  & ((bboxes[:, 2] - bb_widths) < w) \
                  & ((bboxes[:, 3] - bb_heights) < h)

            bboxes = bboxes[ids]
            bboxes = np.clip(bboxes, 0, max(h, w))
            bboxes = bboxes[:, [1, 0, 3, 2]]

            labels = labels[ids]

        if len(labels) == 0:
            labels = np.array([0])
            bboxes = np.array([[0, 0, 1, 1]])

        image_bbox = ImageBBox.create(h, w, bboxes, labels, classes=self.classes, pad_idx=self.pad_idx)
        image_bbox.sample_kwargs = {"domain":self.x.items[i].y[-1]}
        return DomainAdaptationItem(image_bbox)

    def reconstruct(self, t, x):
        (bboxes, labels, domain) = t
        if len((labels - self.pad_idx).nonzero()) == 0: return
        i = (labels - self.pad_idx).nonzero().min()
        bboxes,labels = bboxes[i:],labels[i:]
        return ImageBBox.create(*x.size, bboxes, labels=labels, classes=self.classes, scale=False)

class ObjectItemListSlideDA(SlideObjectItemList):

    def open(self, fn: SlideContainer,  x: int=0, y: int=0):
        return Image(pil2tensor(fn.get_patch(x, y) / 255., np.float32))

class BBBMetrics(LearnerCallback):
    "Add `loss_func.metrics` to metrics named by `loss_func.metric_names`"
    _order = -20 #Needs to run before the recorder

    def on_train_begin(self, **kwargs):
        "Add the metrics names to the `Recorder`."
        self.names = ifnone(self.learn.loss_func.metric_names, [])
        if not self.names: warn('LossMetrics requested by no loss_func.metric_names provided')

        for m in self.learn.metrics:
            if hasattr(m, 'metric_names'):
                for name in m.metric_names:
                    if name not in self.learn.loss_func.metric_names:
                        self.names.append(name)
        if not self.names: warn('Metrics names requested by no metrics.metric_names provided')
        self.learn.recorder.add_metric_names(self.names)

    def on_epoch_begin(self, **kwargs):
        "Initialize the metrics for this epoch."
        self.metrics = {name:0. for name in self.names}
        self.nums = 0

    def on_batch_end(self, last_target, train, **kwargs):
        "Update the metrics if not `train`"
        #return
        if train: return
        bs = last_target[0].size(0)
        for name in self.names:
            if name in self.learn.loss_func.metrics:
                self.metrics[name] += bs * self.learn.loss_func.metrics[name].detach().cpu()
        self.nums += bs

    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if not self.nums: return
        metrics = [self.metrics[name]/self.nums for name in self.names if name in self.learn.loss_func.metrics]

        for name in self.names:
            for metric in self.learn.metrics:
                if hasattr(metric, 'metric_names') and name in metric.metrics.keys():
                    metrics.append(metric.metrics[name])

        return {'last_metrics': last_metrics + metrics}


class PascalVOCMetricByDistanceDAA(PascalVOCMetric):

    def __init__(self, anchors, size, metric_names: list, detect_thresh: float=0.3, nms_thresh: float=0.5
                 , radius: float=25, images_per_batch: int=-1):
        self.ap = 'AP'
        self.anchors = anchors
        self.size = size
        self.detect_thresh = detect_thresh
        self.nms_thresh = nms_thresh
        self.radius = radius

        self.images_per_batch = images_per_batch
        self.metric_names_original = metric_names
        self.metric_names = ["{}-{}".format(self.ap, i) for i in metric_names]

        self.evaluator = Evaluator()
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.removeAllBoundingBoxes()
        self.imageCounter = 0

    def on_epoch_begin(self, **kwargs):
        #print(" Metric Epoch Began")
        self.boundingBoxes.removeAllBoundingBoxes()
        self.imageCounter = 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        #print("Batch Called")
        bbox_gt_batch, class_gt_batch, _ = last_target
        class_pred_batch, bbox_pred_batch = last_output[:2]

        self.images_per_batch = self.images_per_batch if self.images_per_batch > 0 else class_pred_batch.shape[0]
        for bbox_gt, class_gt, clas_pred, bbox_pred in \
                list(zip(bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch))[: self.images_per_batch]:

            bbox_pred, scores, preds = process_output(clas_pred, bbox_pred, self.anchors, self.detect_thresh)
            if bbox_pred is None:# or len(preds) > 3 * len(bbox_gt):
                continue

            #image = np.zeros((512, 512, 3), np.uint8)
            t_sz = torch.Tensor([(self.size, self.size)])[None].cpu()
            bbox_pred = to_np(rescale_boxes(bbox_pred.cpu(), t_sz))
            # change from center to top left
            bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2


            temp_boxes = np.copy(bbox_pred)
            temp_boxes[:, 2] = temp_boxes[:, 0] + temp_boxes[:, 2]
            temp_boxes[:, 3] = temp_boxes[:, 1] + temp_boxes[:, 3]


            to_keep = non_max_suppression_by_distance(temp_boxes, to_np(scores), self.radius, return_ids=True)
            bbox_pred, preds, scores = bbox_pred[to_keep], preds[to_keep].cpu(), scores[to_keep].cpu()

            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0]
            # change gt from x,y,x2,y2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))


            class_gt = to_np(class_gt) - 1
            preds = to_np(preds)
            scores = to_np(scores)

            for box, cla in zip(bbox_gt, class_gt):
                temp = BoundingBox(imageName=str(self.imageCounter), classId=self.metric_names_original[cla], x=box[0], y=box[1],
                               w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(self.size,self.size))

                self.boundingBoxes.addBoundingBox(temp)

            # to reduce math complexity take maximal three times the number of gt boxes
            num_boxes = len(bbox_gt) * 3
            for box, cla, scor in list(zip(bbox_pred, preds, scores))[:num_boxes]:
                temp = BoundingBox(imageName=str(self.imageCounter), classId=self.metric_names_original[cla], x=box[0], y=box[1],
                                   w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute, classConfidence=scor,
                                   bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(self.size, self.size))

                self.boundingBoxes.addBoundingBox(temp)

            #image = self.boundingBoxes.drawAllBoundingBoxes(image, str(self.imageCounter))
            self.imageCounter += 1

    def on_my_end(self, last_output, last_target, **kwargs):
        #print("Batch Called")
        bbox_gt_batch, class_gt_batch, _ = last_target
        class_pred_batch, bbox_pred_batch = last_output[:2]

        self.images_per_batch = self.images_per_batch if self.images_per_batch > 0 else class_pred_batch.shape[0]
        #print(self.images_per_batch)
        res={}
        im_count=-1
        for bbox_gt, class_gt, clas_pred, bbox_pred in list(zip(bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch))[: self.images_per_batch]:
            #print("Image:",im_count)
            im_count+=1
            bbox_pred, scores, preds = process_output(clas_pred, bbox_pred, self.anchors, self.detect_thresh)
            if bbox_pred is None:# or len(preds) > 3 * len(bbox_gt):
                #print("No preds")
                #print(class_gt)
                res[im_count]=[[class_gt.cpu()],[]]
                continue

            #image = np.zeros((512, 512, 3), np.uint8)
            t_sz = torch.Tensor([(self.size, self.size)])[None].cpu()
            bbox_pred = to_np(rescale_boxes(bbox_pred.cpu(), t_sz))
            # change from center to top left
            bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2


            temp_boxes = np.copy(bbox_pred)
            temp_boxes[:, 2] = temp_boxes[:, 0] + temp_boxes[:, 2]
            temp_boxes[:, 3] = temp_boxes[:, 1] + temp_boxes[:, 3]


            to_keep = non_max_suppression_by_distance(temp_boxes, to_np(scores), self.radius, return_ids=True)
            bbox_pred, preds, scores = bbox_pred[to_keep], preds[to_keep].cpu(), scores[to_keep].cpu()

            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0]
            # change gt from x,y,x2,y2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))


            class_gt = to_np(class_gt) - 1
            preds = to_np(preds)
            scores = to_np(scores)

            m_count=0
            r1=[]
            for box, cla in zip(bbox_gt, class_gt):
                #print("Gtbox:",m_count)
                m_count+=1
                temp = BoundingBox(imageName=str(self.imageCounter), classId=self.metric_names_original[cla], x=box[0], y=box[1],
                               w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(self.size,self.size))

                r1.append(temp)
                #self.boundingBoxes.addBoundingBox(temp)

            # to reduce math complexity take maximal three times the number of gt boxes
            num_boxes = len(bbox_gt) * 3
            m_count=0
            r2=[]
            for box, cla, scor in list(zip(bbox_pred, preds, scores))[:num_boxes]:
                #print("Predbox:",m_count)
                m_count+=1
                temp = BoundingBox(imageName=str(self.imageCounter), classId=self.metric_names_original[cla], x=box[0], y=box[1],
                                   w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute, classConfidence=scor,
                                   bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(self.size, self.size))

                r2.append(temp)
                #self.boundingBoxes.addBoundingBox(temp)

            #image = self.boundingBoxes.drawAllBoundingBoxes(image, str(self.imageCounter))
            #self.imageCounter += 1
            res[im_count]=[r1,r2]
        return res
    
    def on_epoch_end(self, last_metrics, **kwargs):
        #print("End of the epoch wrapup")
        if self.boundingBoxes.count() > 0:
            self.metrics = {}
            metricsPerClass = self.evaluator.GetPascalVOCMetrics(self.boundingBoxes, IOUThreshold=0.3)
            self.metric = max(sum([mc[self.ap] for mc in metricsPerClass]) / len(metricsPerClass), 0)

            for mc in metricsPerClass:
                self.metrics['{}-{}'.format(self.ap, mc['class'])] = max(mc[self.ap], 0)

            return {'last_metrics': last_metrics + [self.metric]}
        else:
            self.metrics = dict(zip(self.metric_names, [0 for i in range(len(self.metric_names))]))
            return {'last_metrics': last_metrics + [0]}
    
    def on_man_end(self, **kwargs):
        if self.boundingBoxes.count() > 0:
            self.metrics = {}
            metricsPerClass = self.evaluator.GetPascalVOCMetrics(self.boundingBoxes, IOUThreshold=0.3)
            self.metric = max(sum([mc[self.ap] for mc in metricsPerClass]) / len(metricsPerClass), 0)

            for mc in metricsPerClass:
                self.metrics['{}-{}'.format(self.ap, mc['class'])] = max(mc[self.ap], 0)

            return metricsPerClass
        else:
            self.metrics = dict(zip(self.metric_names, [0 for i in range(len(self.metric_names))]))
            return 0

def bb_pad_collate_da(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    if isinstance(samples[0][1], int): return data_collate(samples)
    max_len = max([len(s[1].data[0][1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    scanner_ids = torch.zeros(len(samples)).long() + pad_idx
    imgs = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        scanner_ids[i] = s[1].scanner_id
        bbs, lbls = s[1].data[0]
        if not (bbs.nelement() == 0):
            bboxes[i,-len(lbls):] = bbs
            labels[i,-len(lbls):] = tensor(lbls)
    return torch.cat(imgs,0), (bboxes,labels,scanner_ids)