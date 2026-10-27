from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def geocoding(string):
    geolocator = Nominatim(user_agent="geocode-draft")

    # Carlisle Park, Flensburg Galerie, Al Noor Markt, Good Luck, Audio Mobile Lab
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
