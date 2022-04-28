import numpy as np

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y
    
def obj(dets, scores, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1e-9) * (y2 - y1 + 1e-9)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        other_box_ids = order[1:]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[other_box_ids])
        yy1 = np.maximum(y1[i], y1[other_box_ids])
        xx2 = np.minimum(x2[i], x2[other_box_ids])
        yy2 = np.minimum(y2[i], y2[other_box_ids])
        

        w = np.maximum(0.0, xx2 - xx1 + 1e-9)
        h = np.maximum(0.0, yy2 - yy1 + 1e-9)
        inter = w * h
          
        ovr = inter / (areas[i] + areas[other_box_ids] - inter)
        
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def get_objects(prediction, conf_thres, iou_thres, top, classes, labels=()):

    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    max_nms = 30000

    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]] 

        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = np.concatenate((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        conf = np.amax(x[:, 5:], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:], axis=1).reshape(conf.shape)
        x = np.concatenate((box, conf, j.astype(float)), axis=1)[conf.flatten() > conf_thres]

        # Filter by class
        if classes is not None:
            print(classes)
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        x = x[x[:, 4].argsort()[::-1][:n]]
        c = x[:, 5:6] * (4096)
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        i = obj(boxes, scores, iou_thres)

        output[xi] = x[i]

    return output
