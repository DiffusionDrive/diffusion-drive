from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def geocoding(string):
    geolocator = Nominatim(user_agent="geocode-draft")

    # Carlisle Park = (54.7761659, 9.4378642)
    # Flensburg Galerie = (54.7832841, 9.4378336)
    # Al Noor Markt = (54.7996017, 9.4229981)
    # Good Luck = (54.8076053, 9.4598226)
    # Audio Mobile Lab = (54.7560858, 9.4021214)
    # Phanomenta = (54.7953852, 9.4301531)
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
