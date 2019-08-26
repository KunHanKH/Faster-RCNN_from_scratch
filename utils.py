import numpy as np

# bboxes_1: anchor
# bboxes_2: target box
def bbox_iou(bboxes_1, bboxes_2):
    len_bboxes_1 = bboxes_1.shape[0]
    len_bboxes_2 = bboxes_2.shape[0]
    ious = np.zeros((len_bboxes_1, len_bboxes_2))

    for idx, bbox_1 in enumerate(bboxes_1):
        yy1_max = np.maximum(bbox_1[0], bboxes_2[:, 0])
        xx1_max = np.maximum(bbox_1[1], bboxes_2[:, 1])
        yy2_min = np.minimum(bbox_1[2], bboxes_2[:, 2])
        xx2_min = np.minimum(bbox_1[3], bboxes_2[:, 3])

        height = np.maximum(0.0, yy2_min - yy1_max)
        width = np.maximum(0.0, xx2_min - xx1_max)

        eps = np.finfo(np.float32).eps
        inter = height * width
        union = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1]) + \
                (bboxes_2[:, 2] - bboxes_2[:, 0]) * (bboxes_2[:, 3] - bboxes_2[:, 1]) - inter + eps
        iou = inter / union
        ious[idx] = iou

    return ious


# (y1, x1, y2, x2) -> (y, x, h, w) -> (dy, dx, dh, dw)
'''
t_{x} = (x - x_{a})/w_{a}
t_{y} = (y - y_{a})/h_{a}
t_{w} = log(w/ w_a)
t_{h} = log(h/ h_a)

anchors are the anchors
base_anchors are the boxes
'''

def format_loc(anchors, base_anchors):
    height = anchors[:, 2] - anchors[:, 0]
    width = anchors[:, 3] - anchors[:, 1]
    ctr_y = anchors[:, 0] + height*0.5
    ctr_x = anchors[:, 1] + width*0.5

    base_height = base_anchors[:, 2] - base_anchors[:, 0]
    base_width = base_anchors[:, 3] - base_anchors[:, 1]
    base_ctr_y = base_anchors[:, 0] + base_height*0.5
    base_ctr_x = base_anchors[:, 1] + base_width*0.5

    eps = np.finfo(np.float32).eps
    height = np.maximum(eps, height)
    width = np.maximum(eps, width)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    anchor_loc_target = np.stack((dy, dx, dh, dw), axis=1)
    return anchor_loc_target


# (dy, dx, dh, dw) -> (y, x, h, w) -> (y1, x1, y2, x2)

'''
anchors are the default anchors
formatted_base_anchors are the boxes with (dy, dx, dh, dw)
'''

def deformat_loc(anchors, formatted_base_anchor):
    height = anchors[:, 2] - anchors[:, 0]
    width = anchors[:, 3] - anchors[:, 1]
    ctr_y = anchors[:, 0] + height*0.5
    ctr_x = anchors[:, 1] + width*0.5
    
    dy, dx, dh, dw = formatted_base_anchor.T
    base_height = np.exp(dh) * height
    base_width = np.exp(dw) * width
    base_ctr_y = dy * height + ctr_y
    base_ctr_x = dx * width + ctr_x
    
    base_anchors = np.zeros_like(anchors)
    base_anchors[:, 0] = base_ctr_y - base_height*0.5
    base_anchors[:, 1] = base_ctr_x - base_width*0.5
    base_anchors[:, 2] = base_ctr_y + base_height*0.5
    base_anchors[:, 3] = base_ctr_x + base_width*0.5
    
    return base_anchors


def nms(rois, scores, nms_thresh):
    order = scores.argsort()[::-1]
    y1, x1, y2, x2 = rois.T
    
    keep_index = []
    
    while order.size > 0:
        i = order[0]
        keep_index.append(i)
        ious = bbox_iou(rois[i][np.newaxis, :], rois[order[1:]])
        inds = np.where(ious <= nms_thresh)[1]
        order = order[inds + 1]
    return keep_index

