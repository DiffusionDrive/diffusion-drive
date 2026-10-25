from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def geocoder_d(string):
    # geolocator = Nominatim(user_agent="geocode-draft")

    # Carlisle Park, Flensburg Galerie, Al Noor Markt, Good Luck, Audio Mobile Lab
    place = string
    city = "Flensburg"
    country = "Germany"

    addr = place + ", " + city + ", " + country

    return addr
    '''
    try:
        location = geolocator.geocode(addr)
        if location:
            print(f"Address: {location.address}")
            print(f"Coordinates: {location.latitude}, {location.longitude}")
        else:
            print("Address not found")

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error: {e}")
    '''