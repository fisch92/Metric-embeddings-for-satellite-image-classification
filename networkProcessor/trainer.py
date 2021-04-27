import json
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import concurrent.futures
import threading
from utils.math import Distances
from utils.converters import Converters
from tensorboardX import SummaryWriter

from dataProcessor.tiffReader import GEOMAP
from network.resnext import resnext50_32x4d
from dataProcessor.miningTypes import MiningTypes
from dataProcessor.batchSampler.poolBatchSampler import PoolBatchSampler
from network.tripletnet import TripletNet
from sklearn.neighbors import KNeighborsClassifier

class Trainer():

	'''
		Trainer schedules training, valdiation and log results to results/name.
		Look at results with tensorboard.
	'''

	def __init__(self,
		imageSampler, 
		batchSampler, 
		net,
		name,
		batch_size,
		ctx=mx.gpu(),
		single_class_validator=None,
		multi_class_validator=None,
		validation_nb=1000,
		validation_map='osm',
		valIteration=1,
		load=True,
		logging=True,
		**kwargs
		):
		self.ctx = ctx
		self.net = net
		self.batchSampler = batchSampler

		#for validation reserved samples
		self.poolBatchSampler = PoolBatchSampler(batch_size, imageSampler, ctx=ctx[0], channels=3)
		self.name = name
		self.batch_size = batch_size

		self.global_step = 0
		self.logger = SummaryWriter(logdir='./results/log'+name)
		self.single_class_validator = single_class_validator
		self.multi_class_validator = multi_class_validator
		self.validation_nb = validation_nb
		self.validation_map = validation_map
		self.valIteration = valIteration
		self.logging = logging
		self.valproc = None

	def resetNet(self):
		self.net.reset()

	def train(self, epochs=100, iterations=1000):
		#lr=0.00002
		for epoch in range(0, epochs):
			score = self.do_epoch(iterations=iterations)
		self.logger.close()
		return score

	def do_epoch(self, iterations=1000):
		
		score = self.validation(int(self.validation_nb/self.batch_size), self.global_step)
		
		if iterations > 0:
			mean_loss = None
			for iteration in range(0, iterations):
				images, coords, px_coords, valid = self.batchSampler.getBatch()
				loss = self.net.train_step(images)
				if mean_loss is None:
					mean_loss = loss
				else:
					mean_loss += loss
				
				self.global_step += 1

			if self.logging:
				self.logger.add_scalars(self.name + 'Loss', {
					'loss': (mean_loss/iterations).mean()
				}, global_step=self.global_step)
			self.net.save()

			overview = None
			for img in images:
				batch = None
				size = 10 if len(img) > 10 else len(img)
				for cbatch in range(0, size):
					cv_img = img[cbatch]
					cv_img = cv_img.transpose((1, 2, 0))

					if batch is None:
						batch = cv_img
					else:
						batch = np.concatenate((batch, cv_img), axis = 1)
				if overview is None:
					overview = batch
				else:
					overview = np.concatenate((overview, batch), axis = 0)
						
			overview = ((overview+1)*127.5).astype('uint8')
			
			print('step: ', self.global_step, 'loss: ', (mean_loss/iterations).mean())
			self.logger.add_image('img', overview[:,:,:3], global_step=self.global_step, dataformats='HWC')


		

		return 0

	def validation(self, iterations=5, global_step=0):
		start = False
		if self.valproc is None: 
			start = True
		elif not self.valproc.is_alive(): #no new validation if valdation is still running
			start = True
		if start:
			if iterations == 0:
				iterations = 1

			
			singleClassScores = {
				'single_accs': {},
				'single_accs_per_class': {},
				'single_f1_scores': {},
				'single_silhouette_scores': {},
				'single_ccc': {}
			}
			multiClassScores = {
				'multi_sum_err': {},
				'multi_nmis': {},
				'gSil': {},
				'multi_ccc': {},
				'multi_local_gSil': {},
				'multi_jsd': {},
				'multi_mapar1': {},
				'multi_mapar5': {},
				'multi_mapar10': {}
			}
			global_embs = []
			global_val_imgs = []
			global_labels = []
			for valiteration in range(0,self.valIteration):
				embs = None
				val_imgs = None
				labels = []
				for iteration in range(0, iterations):
					image, coord, px_coord, valid = self.poolBatchSampler.getBatch()
					
					emb = self.net.predict(image)

					if embs is None:
						embs = emb
					else:
						embs = nd.concat(embs, emb, dim=0)


					if val_imgs is None:
						val_imgs = valid
					else:
						val_imgs = nd.concat(val_imgs, valid, dim=0)


				for i in range(0, len(embs)):
					labels.append('img')

				global_embs.append(embs.asnumpy())
				global_val_imgs.append(val_imgs.asnumpy())
				global_labels.append(labels)

			global_embs_pred = []
			global_val_imgs_pred = []
			global_labels_pred = []
			for valiteration in range(0,self.valIteration):
				embs = None
				val_imgs = None
				imgs = None
				labels = []
				for iteration in range(0, int(500/self.batch_size)):
					image, coord, px_coord, valid = self.poolBatchSampler.getBatch(validation=True)
					emb = self.net.predict(image)

					if embs is None:
						embs = emb
					else:
						embs = nd.concat(embs, emb, dim=0)

					if imgs is None:
						imgs = image
					else:
						imgs = nd.concat(imgs, image, dim=0)

					if val_imgs is None:
						val_imgs = valid
					else:
						val_imgs = nd.concat(val_imgs, valid, dim=0)

				for i in range(0, len(embs)):
					labels.append('img')

				global_embs_pred.append(embs.asnumpy())
				global_val_imgs_pred.append(val_imgs.asnumpy())
				global_labels_pred.append(labels)

			imgs = imgs.asnumpy()

			self.valproc = threading.Thread(target=self.validationEval, args=(global_embs, global_val_imgs, global_labels, global_embs_pred, global_val_imgs_pred, global_labels_pred, singleClassScores, multiClassScores, global_step, imgs))
			self.valproc.start()


	def validationEval(self, global_embs, global_val_imgs, global_labels, global_embs_pred, global_val_imgs_pred, global_labels_pred, singleClassScores, multiClassScores, global_step, imgs):

		for valiteration in range(0,self.valIteration):
			tembs, tval_imgs = (global_embs[valiteration], global_val_imgs[valiteration])
			if not self.single_class_validator is None:
				with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executorSingle:
					futureSingle = executorSingle.submit(self.single_class_validator.train, tembs, tval_imgs)
				
			if not self.multi_class_validator is None:
				with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executorMulti:
					futureMulti = executorMulti.submit(self.multi_class_validator.train, tembs, tval_imgs)

			if not self.single_class_validator is None:
				self.single_class_validator = futureSingle.result()

			if not self.multi_class_validator is None:
				self.multi_class_validator = futureMulti.result()


		for valiteration in range(0,self.valIteration):
			tembs, tval_imgs = (global_embs_pred[valiteration], global_val_imgs_pred[valiteration])

			if not self.multi_class_validator is None:
				with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executorMulti:
					futureMulti = executorMulti.submit(self.multi_class_validation, tembs, tval_imgs, multiClassScores)
				


			if not self.single_class_validator is None:
				with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executorSingle:
					futureSingle = executorSingle.submit(self.single_class_validation, tembs, tval_imgs, singleClassScores)
				accs, accs_per_class, f1_scores, silhouette_scores, ccc, singleClassScores = futureSingle.result()
				#accs, accs_per_class, f1_scores, _, silhouette_scores, _, ccc = self.single_class_validator.scores(embs, val_imgs)

			if not self.multi_class_validator is None:
				
				sum_err, nmis, silhouette_scores, ccc, mapar1, mapar5, mapar10, multiClassScores = futureMulti.result()

				#_, _, sum_err, nmis, silhouette_scores, _, ccc, local_gSil, jsd, mapar1, mapar5, mapar10 = self.multi_class_validator.scores(embs, val_imgs)
				
		if not self.single_class_validator is None:
			
			if self.logging:
				for scorename, score in singleClassScores.items():
					for key, acc in score.items():
						score[key] = np.array(acc).mean().item(), np.array(acc).std().item()
						self.logger.add_scalars(scorename, {
							key: np.array(acc).mean().item()
						}, global_step=global_step)

		print(
			'Single Class: ', '\n',
			json.dumps(singleClassScores, indent=4, sort_keys=True)
			
		)

		if not self.multi_class_validator is None:
			
			if self.logging:
				
				for scorename, score in multiClassScores.items():
					for key, acc in score.items():
						
						score[key] = np.array(acc).mean().item(), np.array(acc).std().item()
						self.logger.add_scalars(scorename, {
							key: np.array(acc).mean().item()
						}, global_step=global_step)
					

		print(
			'Multi Class: ', '\n',
			json.dumps(multiClassScores, indent=4, sort_keys=True)
			
		)
		

		if self.logging:
			#print((imgs[0].asnumpy()+1.)*0.5)
			self.logger.add_embedding(np.concatenate(global_embs[0:2], axis=0)[:1000, :], label_img=(np.concatenate(global_val_imgs[0:2], axis=0)[:1000,:3]+1.)*0.5, global_step=global_step, tag='imgs', metadata=np.concatenate(global_labels[0:2], axis=0)[:1000])
		
	def single_class_validation(self, embs, val_imgs, singleClassScores):
		if not self.single_class_validator is None:
			accs, accs_per_class, f1_scores, silhouette_scores, ccc = self.single_class_validator.scores(embs, val_imgs)

			if self.logging:
				for key, acc in accs.items():
					if not key in singleClassScores['single_accs']:
						singleClassScores['single_accs'][key] = [acc]
					else:
						singleClassScores['single_accs'][key].append(acc)

				for key, acc in accs_per_class.items():
					if not key in singleClassScores['single_accs_per_class']:
						singleClassScores['single_accs_per_class'][key] = [acc]
					else:
						singleClassScores['single_accs_per_class'][key].append(acc)

				for key, acc in f1_scores.items():
					if not key in singleClassScores['single_f1_scores']:
						singleClassScores['single_f1_scores'][key] = [acc]
					else:
						singleClassScores['single_f1_scores'][key].append(acc)

				for key, acc in silhouette_scores.items():
					if not key in singleClassScores['single_silhouette_scores']:
						singleClassScores['single_silhouette_scores'][key] = [acc]
					else:
						singleClassScores['single_silhouette_scores'][key].append(acc)


				for key, acc in ccc.items():
					if not key in singleClassScores['single_ccc']:
						singleClassScores['single_ccc'][key] = [acc]
					else:
						singleClassScores['single_ccc'][key].append(acc)
			
			return accs, accs_per_class, f1_scores, silhouette_scores, ccc, singleClassScores
			
	def multi_class_validation(self, embs, val_imgs, multiClassScores):
		if not self.single_class_validator is None:
			sum_err, nmis, silhouette_scores, ccc, mapar1, mapar5, mapar10 = self.multi_class_validator.scores(embs, val_imgs)
			
			if self.logging:
					
					for key, acc in sum_err.items():
						if not key in multiClassScores['multi_sum_err']:
							multiClassScores['multi_sum_err'][key] = [acc]
						else:
							multiClassScores['multi_sum_err'][key].append(acc)
					for key, acc in nmis.items():
						if not key in multiClassScores['multi_nmis']:
							multiClassScores['multi_nmis'][key] = [acc]
						else:
							multiClassScores['multi_nmis'][key].append(acc)

					for key, acc in silhouette_scores.items():
						if not key in multiClassScores['gSil']:
							multiClassScores['gSil'][key] = [acc]
						else:
							multiClassScores['gSil'][key].append(acc)

					for key, acc in ccc.items():
						if not key in multiClassScores['multi_ccc']:
							multiClassScores['multi_ccc'][key] = [acc]
						else:
							multiClassScores['multi_ccc'][key].append(acc)

					for key, acc in mapar1.items():
						if not key in multiClassScores['multi_mapar1']:
							multiClassScores['multi_mapar1'][key] = [acc]
						else:
							multiClassScores['multi_mapar1'][key].append(acc)
							
					for key, acc in mapar5.items():
						if not key in multiClassScores['multi_mapar5']:
							multiClassScores['multi_mapar5'][key] = [acc]
						else:
							multiClassScores['multi_mapar5'][key].append(acc)
							
					for key, acc in mapar10.items():
						if not key in multiClassScores['multi_mapar10']:
							multiClassScores['multi_mapar10'][key] = [acc]
						else:
							multiClassScores['multi_mapar10'][key].append(acc)

			return sum_err, nmis, silhouette_scores, ccc, mapar1, mapar5, mapar10, multiClassScores