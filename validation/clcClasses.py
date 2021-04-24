from enum import Enum
import numpy as np
from sklearn.neighbors import KDTree

class CLCClasses():


	"""Continuous_urban_fabric = 0
	Discontinuous_urban_fabric = 1
	Industrial_or_commercial_units = 2
	Road_and_rail_networks_and_associated_land = 3
	Port_areas = 4
	Airports = 5
	Mineral_extraction_sites = 6
	Dump_sites = 7
	Construction_sites = 8
	Green_urban_areas = 9
	Sport_and_leisure_facilities = 10
	Non_irrigated_arable_land = 11
	Permanently_irrigated_land = 12
	Rice_fields = 13
	Vineyards = 14
	Fruit_trees_and_berry_plantations = 15
	Olive_groves = 16
	Pastures = 17
	Annual_crops_associated_with_permanent_crops = 18
	Complex_cultivation_patterns = 19
	Land_principally_occupied_by_agriculture = 21
	Agro_forestry_areas = 22
	Broad_leaved_forest = 23
	Coniferous_forest = 24
	Mixed_forest = 25
	Natural_grasslands = 26
	Moors_and_heathland = 27
	Sclerophyllous_vegetation = 28
	Transitional_woodland_shrub = 29
	Beaches_dunes_sands = 20
	Bare_rocks = 30
	Sparsely_vegetated_areas = 31
	Burnt_areas = 32
	Glaciers_and_perpetual_snow = 33
	Inland_marshes = 34
	Peat_bogs = 35
	Salt_marshes = 36
	Salines = 37
	Intertidal_flats = 38
	Water_courses = 39
	Water_bodies = 40
	Coastal_lagoons = 41
	Estuaries = 42
	Sea_and_ocean = 43
	NODATA = 44"""
	labels = np.array([
			[230,0,77],
			[255,0,0],
			[205,77,242],
			[205,0,0],
			[230,205,205],
			[230,205,230],
			[166,0,205],
			[166,77,0],
			[255,77,255],
			[255,166,255],
			[255,230,255],
			[255,255,168],
			[255,255,0],
			[230,230,0],
			[230,128,0],
			[242,166,77],
			[230,166,0],
			[230,230,77],
			[255,230,166],
			[255,230,77],
			[230,205,77],
			[242,205,166],
			[128,255,0],
			[0,166,0],
			[77,255,0],
			[205,242,77],
			[166,255,128],
			[166,230,77],
			[166,242,0],
			[230,230,230],
			[205,205,205],
			[205,255,205],
			[0,0,0],
			[166,230,205],
			[166,166,255],
			[77,77,255],
			[205,205,255],
			[230,230,255],
			[166,166,230],
			[0,205,242],
			[128,242,230],
			[0,255,166],
			[166,255,230],
			[230,242,255],
			[255,255,255]
		])
	tree = KDTree(labels, metric='l1')

	

	def getLabel(color):
		#osm_type = np.argmin(np.abs(color-CLCClasses.labelColors()).mean(axis=1))
		dist, ind = CLCClasses.tree.query(np.expand_dims(color, axis=0), k=1)
		return ind[0][0], dist

	def getLabels(colors):
		#osm_type = np.argmin(np.abs(color-CLCClasses.labelColors()).mean(axis=1))
		dist, ind = CLCClasses.tree.query(colors, k=1)
		return ind[:, 0], dist

		'''if(np.abs(np.array([255,0,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Discontinuous_urban_fabric
			osm_dist = np.abs(np.array([255,0,0])-color).mean()
		if(np.abs(np.array([205,77,242])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Industrial_or_commercial_units
			osm_dist = np.abs(np.array([205,77,242])-color).mean()
		if(np.abs(np.array([205,0,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Road_and_rail_networks_and_associated_land
			osm_dist = np.abs(np.array([205,0,0])-color).mean()
		if(np.abs(np.array([230,205,205])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Port_areas
			osm_dist = np.abs(np.array([230,205,205])-color).mean()
		if(np.abs(np.array([230,205,230])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Airports
			osm_dist = np.abs(np.array([230,205,230])-color).mean()
		if(np.abs(np.array([166,0,205])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Mineral_extraction_sites
			osm_dist = np.abs(np.array([166,0,205])-color).mean()
		if(np.abs(np.array([166,77,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Dump_sites
			osm_dist = np.abs(np.array([166,77,0])-color).mean()
		if(np.abs(np.array([255,77,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Construction_sites
			osm_dist = np.abs(np.array([255,77,255])-color).mean()
		if(np.abs(np.array([255,166,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Green_urban_areas
			osm_dist = np.abs(np.array([255,166,255])-color).mean()
		if(np.abs(np.array([255,230,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Sport_and_leisure_facilities
			osm_dist = np.abs(np.array([255,230,255])-color).mean()
		if(np.abs(np.array([255,255,168])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Non_irrigated_arable_land
			osm_dist = np.abs(np.array([255,255,168])-color).mean()
		if(np.abs(np.array([255,255,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Permanently_irrigated_land
			osm_dist = np.abs(np.array([255,255,0])-color).mean()
		if(np.abs(np.array([230,230,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Rice_fields
			osm_dist = np.abs(np.array([230,230,0])-color).mean()

		if(np.abs(np.array([230,128,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Vineyards
			osm_dist = np.abs(np.array([230,128,0])-color).mean()
		if(np.abs(np.array([242,166,77])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Fruit_trees_and_berry_plantations
			osm_dist = np.abs(np.array([242,166,77])-color).mean()
		if(np.abs(np.array([230,166,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Olive_groves
			osm_dist = np.abs(np.array([230,166,0])-color).mean()
		if(np.abs(np.array([230,230,77])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Pastures
			osm_dist = np.abs(np.array([230,230,77])-color).mean()
		if(np.abs(np.array([255,230,166])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Annual_crops_associated_with_permanent_crops
			osm_dist = np.abs(np.array([255,230,166])-color).mean()
		if(np.abs(np.array([255,230,77])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Complex_cultivation_patterns
			osm_dist = np.abs(np.array([255,230,77])-color).mean()
		if(np.abs(np.array([230,205,77])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Land_principally_occupied_by_agriculture
			osm_dist = np.abs(np.array([230,205,77])-color).mean()
		if(np.abs(np.array([242,205,166])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Agro_forestry_areas
			osm_dist = np.abs(np.array([242,205,166])-color).mean()
		if(np.abs(np.array([128,255,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Broad_leaved_forest
			osm_dist = np.abs(np.array([128,255,0])-color).mean()
		if(np.abs(np.array([0,166,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Coniferous_forest
			osm_dist = np.abs(np.array([0,166,0])-color).mean()
		if(np.abs(np.array([77,255,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Mixed_forest
			osm_dist = np.abs(np.array([77,255,0])-color).mean()
		if(np.abs(np.array([205,242,77])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Natural_grasslands
			osm_dist = np.abs(np.array([205,242,77])-color).mean()

		if(np.abs(np.array([166,255,128])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Moors_and_heathland
			osm_dist = np.abs(np.array([166,255,128])-color).mean()
		if(np.abs(np.array([166,230,77])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Sclerophyllous_vegetation
			osm_dist = np.abs(np.array([166,230,77])-color).mean()
		if(np.abs(np.array([166,242,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Transitional_woodland_shrub
			osm_dist = np.abs(np.array([166,242,0])-color).mean()
		if(np.abs(np.array([230,230,230])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Beaches_dunes_sands
			osm_dist = np.abs(np.array([230,230,230])-color).mean()
		if(np.abs(np.array([205,205,205])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Bare_rocks
			osm_dist = np.abs(np.array([205,205,205])-color).mean()
		if(np.abs(np.array([205,255,205])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Sparsely_vegetated_areas
			osm_dist = np.abs(np.array([205,255,205])-color).mean()
		if(np.abs(np.array([0,0,0])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Burnt_areas
			osm_dist = np.abs(np.array([0,0,0])-color).mean()
		if(np.abs(np.array([166,230,205])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Glaciers_and_perpetual_snow
			osm_dist = np.abs(np.array([166,230,205])-color).mean()
		if(np.abs(np.array([166,166,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Inland_marshes
			osm_dist = np.abs(np.array([166,166,255])-color).mean()
		if(np.abs(np.array([77,77,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Peat_bogs
			osm_dist = np.abs(np.array([77,77,255])-color).mean()
		if(np.abs(np.array([205,205,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Salt_marshes
			osm_dist = np.abs(np.array([205,205,255])-color).mean()
		if(np.abs(np.array([230,230,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Salines
			osm_dist = np.abs(np.array([230,230,255])-color).mean()

		if(np.abs(np.array([166,166,230])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Intertidal_flats
			osm_dist = np.abs(np.array([166,166,230])-color).mean()
		if(np.abs(np.array([0,205,242])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Water_courses
			osm_dist = np.abs(np.array([0,205,242])-color).mean()
		if(np.abs(np.array([128,242,230])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Water_bodies
			osm_dist = np.abs(np.array([128,242,230])-color).mean()
		if(np.abs(np.array([0,255,166])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Coastal_lagoons
			osm_dist = np.abs(np.array([0,255,166])-color).mean()
		if(np.abs(np.array([166,255,230])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Estuaries
			osm_dist = np.abs(np.array([166,255,230])-color).mean()
		if(np.abs(np.array([230,242,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.Sea_and_ocean
			osm_dist = np.abs(np.array([230,242,255])-color).mean()
		if(np.abs(np.array([255,255,255])-color).mean() < osm_dist):
			osm_type =  CLCClasses.NODATA
			osm_dist = np.abs(np.array([255,255,255])-color).mean()

		return osm_type.value'''

	def getColor(osm_type):
		return CLCClasses.labels[osm_type]
		"""osm_type = CLCClasses(osm_type)
		if osm_type == CLCClasses.Continuous_urban_fabric:
			return np.array([230,0,77])

		if osm_type ==  CLCClasses.Discontinuous_urban_fabric:
			return np.array([255,0,0])
		if osm_type ==  CLCClasses.Industrial_or_commercial_units:
			return np.array([205,77,242])
		if osm_type ==  CLCClasses.Road_and_rail_networks_and_associated_land:
			return np.array([205,0,0])
		if osm_type ==  CLCClasses.Port_areas:
			return np.array([230,205,205])
		if osm_type ==  CLCClasses.Airports:
			return np.array([230,205,230])
		if osm_type ==  CLCClasses.Mineral_extraction_sites:
			return np.array([166,0,205])
		if osm_type ==  CLCClasses.Dump_sites:
			return np.array([166,77,0])
		if osm_type ==  CLCClasses.Construction_sites:
			return np.array([255,77,255])
		if osm_type ==  CLCClasses.Green_urban_areas:
			return np.array([255,166,255])
		if osm_type ==  CLCClasses.Sport_and_leisure_facilities:
			return np.array([255,230,255])
		if osm_type ==  CLCClasses.Non_irrigated_arable_land:
			return np.array([255,255,168])
		if osm_type ==  CLCClasses.Permanently_irrigated_land:
			return np.array([255,255,0])
		if osm_type ==  CLCClasses.Rice_fields:
			return np.array([230,230,0])

		if osm_type ==  CLCClasses.Vineyards:
			return np.array([230,128,0])
		if osm_type ==  CLCClasses.Fruit_trees_and_berry_plantations:
			return np.array([242,166,77])
		if osm_type ==  CLCClasses.Olive_groves:
			return np.array([230,166,0])
		if osm_type ==  CLCClasses.Pastures:
			return np.array([230,230,77])
		if osm_type ==  CLCClasses.Annual_crops_associated_with_permanent_crops:
			return np.array([255,230,166])
		if osm_type ==  CLCClasses.Complex_cultivation_patterns:
			return np.array([255,230,77])
		if osm_type ==  CLCClasses.Land_principally_occupied_by_agriculture:
			return np.array([230,205,77])
		if osm_type ==  CLCClasses.Agro_forestry_areas:
			return np.array([242,205,166])
		if osm_type ==  CLCClasses.Broad_leaved_forest:
			return np.array([128,255,0])
		if osm_type ==  CLCClasses.Coniferous_forest:
			return np.array([0,166,0])
		if osm_type ==  CLCClasses.Mixed_forest:
			return np.array([77,255,0])
		if osm_type ==  CLCClasses.Natural_grasslands:
			return np.array([205,242,77])

		if osm_type ==  CLCClasses.Moors_and_heathland:
			return np.array([166,255,128])
		if osm_type ==  CLCClasses.Sclerophyllous_vegetation:
			return np.array([166,230,77])
		if osm_type ==  CLCClasses.Transitional_woodland_shrub:
			return np.array([166,242,0])
		if osm_type ==  CLCClasses.Beaches_dunes_sands:
			return np.array([230,230,230])
		if osm_type ==  CLCClasses.Bare_rocks:
			return np.array([205,205,205])
		if osm_type ==  CLCClasses.Sparsely_vegetated_areas:
			return np.array([205,255,205])
		if osm_type ==  CLCClasses.Burnt_areas:
			return np.array([0,0,0])-color	
		if osm_type ==  CLCClasses.Glaciers_and_perpetual_snow:
			return np.array([166,230,205])
		if osm_type ==  CLCClasses.Inland_marshes:
			return np.array([166,166,255])
		if osm_type ==  CLCClasses.Peat_bogs:
			return np.array([77,77,255])
		if osm_type ==  CLCClasses.Salt_marshes:
			return np.array([205,205,255])
		if osm_type ==  CLCClasses.Salines:
			return np.array([230,230,255])

		if osm_type ==  CLCClasses.Intertidal_flats:
			return np.array([166,166,230])
		if osm_type ==  CLCClasses.Water_courses:
			return np.array([0,205,242])
		if osm_type ==  CLCClasses.Water_bodies:
			return np.array([128,242,230])
		if osm_type ==  CLCClasses.Coastal_lagoons:
			return np.array([0,255,166])
		if osm_type ==  CLCClasses.Estuaries:
			return np.array([166,255,230])
		if osm_type ==  CLCClasses.Sea_and_ocean:
			return np.array([230,242,255])
		if osm_type ==  CLCClasses.NODATA:
			return np.array([255,255,255])"""