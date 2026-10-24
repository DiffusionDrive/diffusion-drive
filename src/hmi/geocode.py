from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

geolocator = Nominatim(user_agent="geocode-draft")

# Carlisle Park, Flensburg Galerie, Al Noor Markt, Good Luck, Audio Mobile Lab

try:
    location = geolocator.geocode("Audio Mobile Lab, Flensburg, Germany")
    if location:
        print(f"Address: {location.address}")
        print(f"Coordinates: {location.latitude}, {location.longitude}")
    else:
        print("Address not found")

except (GeocoderTimedOut, GeocoderServiceError) as e:
    print(f"Geocoding error: {e}")
