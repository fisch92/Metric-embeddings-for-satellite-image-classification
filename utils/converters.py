import cv2
import numpy as np


class Converters():

	def prepare_embeddings_for_logging(embtuple, imgtuple, labels):
		out_imgs = None
		out_label = []
		for imgs in range(0, len(imgtuple)):
			for img in imgtuple[imgs]:
				img = (img).asnumpy().transpose((1, 2, 0))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = np.expand_dims(img, axis=0)
				img = img.transpose((0, 3, 1, 2))
				if out_imgs is None:
					out_imgs = img
				else:
					out_imgs = np.concatenate((out_imgs, img), axis=0)
				out_label.append(labels[imgs])
		out_embs = None
		for emb in embtuple:
			emb = emb.asnumpy()
			if out_embs is None:
				out_embs = emb
			else:
				out_embs = np.concatenate((out_embs, emb), axis=0)
		return out_embs, out_imgs, out_label
		