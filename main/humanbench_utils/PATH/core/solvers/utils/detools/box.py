import cv2
import numpy
# from IPython import embed
import random
import pdb
class BoxBase(object):
    """
    :class: basic bounding box class
    :ivar float x: left boundary coordinate of bounding box
    :ivar float y: top boundary coordinate of bounding box
    :ivar float w: width of bounding box
    :ivar float h: height of bounding box
    :ivar str tag: tag of bounding box
    """
    def __init__(self, x=0.0, y=0.0, w=0.0, h=0.0, tag=None):
        self.x, self.y, self.w, self.h = map(float, [x, y, w, h])
        self.tag = tag
        self.eps = 1e-6

    def __str__(self):
        return str(self.dumpOdf())

    @property
    def x1(self):
        """
        :return: right boundary coordinate of bounding box
        """
        return self.x + self.w

    @property
    def y1(self):
        """
        :return: bottom boundary coordinate of bounding box
        """
        return self.y + self.h

    @property
    def cx(self):
        """
        :return: center x coordinate of bounding box
        """
        return self.x + self.w / 2.0

    @property
    def cy(self):
        """
        :return: center y coordinate of bounding box
        """
        return self.y + self.h / 2.0

    @property
    def area(self):
        """
        :return: area of bounding box
        """
        return self.w * self.h

    def parseOdf(self, odf):
        """
        :meth: read the object from a dict
        :param odf: a dict with "box" key, e.g., odf = {"box": [5,5,20,50]}
        :type odf: dict
        """
        if "box" in odf:
            self.x, self.y, self.w, self.h = odf["box"]
        if "tag" in odf:
            self.tag = odf["tag"]
    def parseOdfbyname(self,odf,name):
        if name in odf:
            self.x,self.y,self.w,self.h = odf[name]
        if "tag" in odf:
            self.tag = odf["tag"]
    def dumpOdf(self):
        """
        :meth: dump the object into a dict
        :return: a dict with "box" key
        """
        odf = dict()
        odf["box"] = [self.x, self.y, self.w, self.h]
        if self.tag is not None:
            odf["tag"] = self.tag
        return odf

    def parseNp(self, arr):
        """
        :meth: parse from numpy array
        :param arr: a numpy array in the format of [x, y, x1, y1]
        :type arr: numpy.array
        """
        self.x, self.y, self.w, self.h = map(float, \
                [arr[0], arr[1], (arr[2]-arr[0]), (arr[3]-arr[1])])

    def dumpNp(self):
        """
        :meth: dump the object into a numpy array
        :return: a numpy array
        """
        return numpy.array([self.x, self.y, self.x1, self.y1], dtype="float32")

    def intersection(self, boxB):
        """
        :param boxB: the target bounding box to compute overlap
        :type boxB: BoxBase
        :return: the intersection area of current bounding box and boxB
        """
        s1 = (self.y1 - self.y) * (self.x1 - self.x)
        s2 = (boxB.y1 - boxB.y) * (boxB.x1 - boxB.x)
        s0 = 0.0 #intersection area
        if not(self.x >= boxB.x1 or boxB.x >= self.x1 or self.y >= boxB.y1 or boxB.y >= self.y1):
            x = max(self.x, boxB.x); x1 = min(self.x1, boxB.x1)
            y = max(self.y, boxB.y); y1 = min(self.y1, boxB.y1)
            s0 = abs(x - x1) * abs(y - y1)
        return float(s0)

    def iou(self, boxB):
        """
        :param boxB: the target bounding box to compute overlap
        :type boxB: BoxBase
        :return: the intersection area divided by the union area of current bounding box and boxB
        """
        s0 = self.intersection(boxB)
        s = self.area + boxB.area - s0 #union area
        return s0 / float(s+self.eps)

    def ioa(self, boxB):
        """
        :param boxB: the target bounding box to compute overlap
        :type boxB: BoxBase
        :return: the intersection area divided by the area of current bounding box
        """
        return self.intersection(boxB) / float(self.area+self.eps)

    def iob(self, boxB):
        """
        :param boxB: the target bounding box to compute overlap
        :type boxB: BoxBase
        :return: the intersection area divided by the area of boxB
        """
        return self.intersection(boxB) / float(boxB.area+self.eps)

    def iomin(self, boxB):
        """
        :param boxB: the target bounding box to compute overlap
        :type boxB: BoxBase
        :return: the intersection area divided by the min area of current bounding box and boxB
        """
        return self.intersection(boxB) / float(min(self.area, boxB.area)+self.eps)

    def _getDrawColor(self):
        """
        :meth: get color for drawing bounding box
        :return: 3-D tuple as color, default white as (255,255,255)
        """
        return (255, 255, 255)

    def _getDrawLine(self, bold=False):
        """
        :meth: get line size for drawing bounding box
        :param bold: if bold the bounding box or not, default False
        :type bold: bool
        """
        return 2 if bold else 1

    def draw(self, img, bold=False):
        """
        :meth: draw bounding box on the given image
        :param img: the image to draw
        :param bold: if bold the bounding box or not (default False)
        :type img: numpy.array
        :type bold: bool
        """
        color = self._getDrawColor()
        line = self._getDrawLine(bold)
        if "score" in self.__dict__:
            text = "{}: {:.4f}".format(self.tag, self.score)
        else:
            text = str(self.tag)

        cv2.rectangle(img, (int(self.x), int(self.y)), (int(self.x1), int(self.y1)), color, line)
        cx = self.x + (self.x1 - self.x) / 2 - 5
        cy = self.y + 12
        cv2.putText(img, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        """
        cv2.putText(img, text, (int(self.x1)+5, int(self.y1)+5), cv2.FONT_HERSHEY_SIMPLEX,\
                    0.5, color, line)
        """

    def draw_lzm(self, img, colors_lzm, bold=False):
        """
        :meth: draw bounding box on the given image
        :param img: the image to draw
        :param bold: if bold the bounding box or not (default False)
        :type img: numpy.array
        :type bold: bool
        """
        if self.tag not in colors_lzm:
            colors_lzm[self.tag] = (
                        random.random() * 255, random.random() * 255,
                        random.random() * 255)
        color = colors_lzm[self.tag]
        # line = self._getDrawLine(bold)
        if "score" in self.__dict__:
            text = "{}: {:.2f}".format(self.tag, self.score)
        else:
            text = str(self.tag)

        cv2.rectangle(img, (int(self.x), int(self.y)), (int(self.x1), int(self.y1)), color, 3)
        cx = self.x
        cy = self.y - 12
        #cv2.putText(img, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        cv2.putText(img, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
        """
        cv2.putText(img, text, (int(self.x1)+5, int(self.y1)+5), cv2.FONT_HERSHEY_SIMPLEX,\
                    0.5, color, line)
        """

class DetBox(BoxBase):
    """
    :class: bounding box for detection result, inherited from BoxBase
    :ivar float score: detection score (for one class) of bounding box
    :ivar int matched: if matched with a groundtruth. 0 for unmatched (default); 1 for matched; -1 for matched with an igonred groundtruth
    """
    def __init__(self, x=0.0, y=0.0, w=0.0, h=0.0, tag=None, score=0.0, color=None):
        super(DetBox, self).__init__(x, y, w, h, tag)
        self.score = score
        self.matched = 0 # 0 for unmatched / 1 for matched / -1 for matched with ignored gt
        self.color = color

    def parseOdf(self, odf):
        """
        :meth: read the object from a dict
        :param dict odf: a dict with "box" and "score" keys, e.g., odf = {"box": [5,5,20,50], "score": 0.4}
        """
        super(DetBox, self).parseOdf(odf)
        if "score" in odf:
            self.score = odf["score"]
        self.matched = 0
    def parseOdfbyname(self,odf,name):

        super(DetBox,self).parseOdfbyname(odf,name)
        if "score" in odf:
            self.score = odf["score"]
        self.matched = 0
    def dumpOdf(self):
        """
        :meth: dump the object into a dict
        :return: a dict with "box" and "score" keys
        """
        odf = super(DetBox, self).dumpOdf()
        odf["score"] = self.score
        return odf

    def _getDrawColor(self):
        """
        :meth: get the color for detection box. Blue for unmatched (self.matched=0); green for matched (self.matched=1); white for matched with ignored groundtruth (self.matched=-1)
        """
        if self.color is not None:
            return self.color
        color = (0, 255, 0) # green for matched dt (unignored)
        """
        if self.matched == 0:
            color = (255, 0, 0) # blue for unmatched dt
        elif self.matched == 1:
            color = (0, 255, 0) # green for matched dt (unignored)
        elif self.matched == -1:
            color = (255, 255, 255) # white for matched dt (ignored)
        """
        return color


class DetBoxGT(BoxBase):
    """
    :class: bounding box for detection groundtruth, inherited from BoxBase
    :ivar int ign: if the bounding box should be ignored or not. 0 for NOT ignored (default); 1 for ignored.
    :ivar float occ: occlusion rate of the bounding box (default 0.0), e.g., occ=0.6 means 60% of the box is invisible
    :ivar int matched: if matched with a detection result. 0 for unmatched (default); 1 for matched
    """
    def __init__(self, x=0.0, y=0.0, w=0.0, h=0.0, tag=None, ign=0, occ=0.0, color=None):
        super(DetBoxGT, self).__init__(x, y, w, h, tag)
        self.ign = ign
        self.occ = occ
        self.matched = 0 # 0 for unmatched / 1 for matched
        self.color = color

    def parseOdf(self, odf):
        """
        :meth: read the object from a dict
        :param dict odf: a dict with "box", "score", "occ" and "extra.ignore" keys, e.g., odf = {"box": [5,5,20,50], "tag": "person", "occ": 0.6, "extra": {"ignore": 0}}
        """
        super(DetBoxGT, self).parseOdf(odf)
        if "extra" in odf and "ignore" in odf["extra"]:
            self.ign = odf["extra"]["ignore"]
        if "occ" in odf:
            self.occ = odf["occ"]
        self.matched = 0
    def parseOdfbyname(self,odf,name):
        #pdb.set_trace()
        #params = {'odf':odf,'name':name}
        super(DetBoxGT,self).parseOdfbyname(odf,name)
        if "extra" in odf and "ignore" in odf["extra"]:
            self.ign = odf["extra"]["ignore"]
        if "occ" in odf:
            self.occ = odf["occ"]
        self.matched = 0

    def dumpOdf(self):
        """
        :meth: dump the object into a dict
        :return: a dict with "box", "score", "occ" and "extra.ignore" keys
        """
        odf = super(DetBoxGT, self).dumpOdf()
        odf["extra"] = {"ignore": self.ign}
        odf["occ"] = self.occ
        return odf

    def _getDrawColor(self):
        """
        :meth: get the color for groundtruth box. Red for unmatched (self.matched=0 & self.ign=0); cyan for matched (self.matched=1 & self.ign=0); yellow for ignored groundtruth (self.ign=1)
        """
        if self.color is not None:
            return self.color
        if self.ign == 1:
            color = (0, 255, 255) # yellow for ignored gt
        elif self.matched == 0:
            color = (0, 0, 255) # red for unmatched gt
        elif self.matched == 1:
            color = (255, 255, 0) # cyan for matched gt
        return color

class BoxUtil:
    """
    :class: Implements some common utils for boxes parsing, transformation
    """
    @classmethod
    def draw_img_boxes(cls, img, boxes, t, color_dict=None, bold=False):
        """
        :t: type, 'pd' for prediction box, 'gt' for ground truth box
        """
        for box in boxes:
            if t == 'pd':
                draw_box = DetBox()
            else:
                draw_box = DetBoxGT()
            if type(box) == dict:
                draw_box.parseOdf(box)
            else:
                draw_box.parseNp(box)
            if color_dict is not None:
                draw_box.color = color_dict.get(draw_box.tag)
            draw_box.draw(img, bold=bold)
        return img

    @classmethod
    def parse_gt_boxes(cls, gtboxes):
        rt_boxes = []
        for box in gtboxes:
            gt_box = DetBoxGT()
            gt_box.parseOdf(box)
            rt_boxes.append(gt_box)
        return rt_boxes

    @classmethod
    def parse_gt_boxes_by_name(cls, gtboxes,name):
        rt_boxes = []
        for box in gtboxes:
            import pdb
            #pdb.set_trace()
            gt_box = DetBoxGT()
            gt_box.parseOdfbyname(box,name)
            rt_boxes.append(gt_box)
        return rt_boxes
