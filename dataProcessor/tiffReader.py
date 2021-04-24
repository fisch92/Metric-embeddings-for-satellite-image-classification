import rasterio
import cv2
import math
from enum import Enum
import numpy as np
import scipy as sc
import mxnet.ndarray as nd
from rasterio.windows import Window
from numpy.polynomial import polynomial as P
from numpy.polynomial.polynomial import polyval
from scipy.interpolate import Rbf

OSMUPPERLEFTLON = 4.9407341
OSMUPPERLEFTLAN = 55.7054442
OSMLOWERLEFTLON = 4.9407341
OSMLOWERLEFTLAN = 47.3537047

OSMUPPERRIGHTLON = 18.4154633
OSMUPPERRIGHTLAN = 55.7054442
OSMLOWERRIGHTLON = 18.4154633
OSMLOWERRIGHTLAN = 47.3537047

S2UPPERLEFTLON = 5.6666654
S2UPPERLEFTLAN = 56.0000020
S2LOWERLEFTLON = 5.6666654
S2LOWERLEFTLAN = 46.4000005

S2UPPERRIGHTLON = 15.3333306
S2UPPERRIGHTLAN = 56.0000020
S2LOWERRIGHTLON = 15.3333306
S2LOWERRIGHTLAN = 46.4000005

class MissingDataError(Exception):
	pass

class GEOMAP(Enum):
	SENTINEL = './data/s2-de.tif'
	OSM = './data/osm_WGS84.tif'
	CLC = './data/CLCBGR.tif'

	def toString(geomap):
		if geomap == GEOMAP.SENTINEL:
			return 'sentinel'
		if geomap == GEOMAP.OSM:
			return 'osm'
		if geomap == GEOMAP.CLC:
			return 'clc'

class TiffReader():

	def __init__(self, geomap=GEOMAP.SENTINEL):
		
		self.geomap = geomap
		with rasterio.open(self.geomap.value) as geomap_file:
			self.width = geomap_file.width
			self.height = geomap_file.height
			self.transform = geomap_file.transform

		

	def coord2Px(self, lon, lan):
		if type(lon) is nd.NDArray:
			lon = lon.asscalar()
		if type(lan) is nd.NDArray:
			lan = lan.asscalar()
		with rasterio.open(self.geomap.value) as geomap:
			y, x = geomap.index(lon, lan)
		return x, y

	def px2Coord(self, x, y):
		if type(x) is nd.NDArray:
			x = x.asscalar()
		if type(y) is nd.NDArray:
			y = y.asscalar()
		with rasterio.open(self.geomap.value) as geomap:
			lon, lan = geomap.xy(y, x)
		
		return lon, lan



	def readTilePx(self, x, y, width, height):
		with rasterio.open(self.geomap.value) as geomap:
			#print(geomap.get_transform())
			if type(x) is nd.NDArray:
				x = x.asscalar()
			if type(y) is nd.NDArray:
				y = y.asscalar()

			tile = geomap.read(window=Window(math.floor(x), math.floor(y), width, height))
			#print(tile.shape)
			if tile.shape[1] != int(height) or tile.shape[2] != int(width):
				raise MissingDataError("Can't read tile! Coord: ", x, y, "Max X: ", self.width, "Max Y: ", self.height, "Shape: ", tile.shape, "Size: ", width, height)
			return tile[0:3]

	def readTileCoord(self, lon, lan, width, height):
		with rasterio.open(self.geomap.value) as geomap:

			if type(lon) is nd.NDArray:
				lon = lon.asscalar()
			if type(lan) is nd.NDArray:
				lan = lan.asscalar()
			y, x = geomap.index(lon, lan)
			tile = geomap.read(window=Window(int(x), int(y), width, height))
			#print(tile.shape)
			if tile.shape[1] != int(height) or tile.shape[2] != int(width):
				raise MissingDataError("Can't read tile! Coord: ", x, y, "Max X: ", self.width, "Max Y: ", self.height, "Shape: ", tile.shape, "Size: ", width, height)
			return tile[0:3]

def unitTest():
	
	image = None
	size = 512+1024
	lan,lon = 50.702329, 10.896561
	
	tiffs2 = TiffReader(geomap=GEOMAP.SENTINEL)
	tiffosm = TiffReader(geomap=GEOMAP.CLC)
	tiles2 = tiffs2.readTileCoord( lon, lan, size, size)
	tiles2 = tiles2.transpose((1, 2, 0))
	x, y = tiffs2.coord2Px(lon, lan)
	end_lon, end_lan = tiffs2.px2Coord(x+size,y+size)

	
	start_x, start_y = tiffosm.coord2Px(lon,lan)
	end_x, end_y = tiffosm.coord2Px(end_lon,end_lan)
	print(start_x, start_y, end_x, end_y)
	tileosm = tiffosm.readTilePx( start_x, start_y, abs(end_x-start_x), abs(end_y-start_y))
	tileosm = tileosm.transpose((1, 2, 0))
	
	print(tileosm.shape)
	tileosm = cv2.resize(tileosm, (size, size))
	print(tiles2.shape, tileosm.shape)
	out = cv2.addWeighted(tiles2, 0.7, tileosm, 0.3, 0)
	tileosm = cv2.cvtColor(tileosm, cv2.COLOR_BGRA2RGB)
	print(tileosm)
	cv2.imshow('sample image',out)
 
	cv2.waitKey(0) # waits until a key is pressed
	cv2.destroyAllWindows() # destroys the window showing image
	#print('offset', (start_lon*2048, start_lan*2048),image.shape)

if __name__ == '__main__':
	unitTest()
		
