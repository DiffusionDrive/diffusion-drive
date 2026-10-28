from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def geocoding(string):
    geolocator = Nominatim(user_agent="geocode-draft")

    # Flensburg Galerie = (54.7832841, 9.4378336),
    # Equippers Flensburg, Alter Friedhof
    # Carlisle Park (south) = (54.7761659, 9.4378642)
    # Al Noor Markt (north-west) = (54.7996017, 9.4229981)
    # Good Luck (north-east) = (54.8076053, 9.4598226)
    # Cafe Riebe (north-east) = (54.7979483, 9.4529999)
    # Sankt Jurgen (north) = (54.7910565, 9.4416068)
    # Kapelle (west) = (54.7842927, 9.4215099)
    # Phanomenta (north-west) = (54.7953852, 9.4301531)
    place = string
    city = "Flensburg"
    country = "Germany"

    addr = place + ", " + city + ", " + country

    try:
        location = geolocator.geocode(addr)
        if not location:
            print("Address not found")

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error: {e}")

    return location.latitude, location.longitude
