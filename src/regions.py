"""Collection boxes only; these are NOT official war-risk boundaries."""
REGIONS = [
 {"id":"hormuz","name":"Strait of Hormuz","south":23.0,"west":52.0,"north":28.5,"east":59.0},
 {"id":"malacca","name":"Strait of Malacca","south":-1.5,"west":95.0,"north":7.0,"east":104.5},
 {"id":"bab_el_mandeb","name":"Bab el-Mandeb / southern Red Sea","south":11.0,"west":40.0,"north":16.5,"east":45.5},
 {"id":"suez","name":"Suez / northern Red Sea","south":27.0,"west":31.0,"north":33.5,"east":36.5},
 {"id":"panama","name":"Panama Canal approaches","south":7.0,"west":-83.5,"north":10.5,"east":-78.0},
 {"id":"taiwan","name":"Taiwan Strait","south":21.0,"west":117.0,"north":27.5,"east":122.5},
 {"id":"gibraltar","name":"Strait of Gibraltar","south":34.0,"west":-7.0,"north":37.5,"east":-4.0},
 {"id":"turkish","name":"Turkish Straits approaches","south":39.0,"west":25.0,"north":42.5,"east":31.5},
]
def aisstream_boxes():
    return [[[r['south'],r['west']],[r['north'],r['east']]] for r in REGIONS]
def region_for(lat,lon):
    for r in REGIONS:
        if r['south'] <= lat <= r['north'] and r['west'] <= lon <= r['east']:
            return r['name']
    return 'Other'
