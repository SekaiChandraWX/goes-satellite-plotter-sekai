import streamlit as st
import os
import requests
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import tempfile
from geopy.geocoders import Nominatim
import time
import xml.etree.ElementTree as ET

# Set page config
st.set_page_config(
    page_title="GOES Satellite Viewer",
    page_icon="🛰️",
    layout="wide"
)

# Satellite configurations
SATELLITES = {
    'GOES-16': {
        'longitude': -75.2,
        'start_date': datetime(2017, 4, 1, 0, 0),
        'end_date': datetime(2024, 10, 10, 20, 0),
        'bucket': 'noaa-goes16',
        'identifier': 'G16'
    },
    'GOES-17': {
        'longitude': -137.0,
        'start_date': datetime(2018, 8, 28, 0, 0),
        'end_date': datetime(2022, 7, 27, 23, 59),
        'bucket': 'noaa-goes17',
        'identifier': 'G17'
    },
    'GOES-18': {
        'longitude': -137.0,
        'start_date': datetime(2022, 7, 28, 0, 0),
        'end_date': datetime(2030, 12, 31, 23, 59),  # Future end date
        'bucket': 'noaa-goes18',
        'identifier': 'G18'
    },
    'GOES-19': {
        'longitude': -75.2,
        'start_date': datetime(2024, 10, 10, 20, 0),
        'end_date': datetime(2030, 12, 31, 23, 59),  # Future end date
        'bucket': 'noaa-goes19',
        'identifier': 'G19'
    }
}


def rbtop3():
    """Custom colormap for temperature visualization"""
    newcmp = mcolors.LinearSegmentedColormap.from_list("", [
        (0 / 140, "#000000"),
        (60 / 140, "#fffdfd"),
        (60 / 140, "#05fcfe"),
        (70 / 140, "#010071"),
        (80 / 140, "#00fe24"),
        (90 / 140, "#fbff2d"),
        (100 / 140, "#fd1917"),
        (110 / 140, "#000300"),
        (120 / 140, "#e1e4e5"),
        (120 / 140, "#eb6fc0"),
        (130 / 140, "#9b1f94"),
        (140 / 140, "#330f2f")
    ])
    return newcmp.reversed()


def get_coordinates(location_str):
    """Geocode location string to lat/lon coordinates"""
    try:
        geolocator = Nominatim(user_agent="goes_satellite_viewer")
        time.sleep(1)  # Rate limiting
        location = geolocator.geocode(location_str, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return None, None


def select_best_satellite(lon, lat, requested_date):
    """Select the best satellite based on location and date"""
    # Calculate distance to each satellite longitude
    distances = {}
    available_satellites = []

    for sat_name, sat_info in SATELLITES.items():
        if sat_info['start_date'] <= requested_date <= sat_info['end_date']:
            available_satellites.append(sat_name)
            # Calculate angular distance (simplified)
            distance = abs(lon - sat_info['longitude'])
            if distance > 180:
                distance = 360 - distance
            distances[sat_name] = distance

    if not available_satellites:
        return None, "No satellite data available for the requested date"

    # Select satellite with minimum distance
    best_satellite = min(available_satellites, key=lambda x: distances[x])
    return best_satellite, None


def list_files(bucket_url):
    """List files in AWS S3 bucket"""
    try:
        response = requests.get(bucket_url, timeout=30)
        if response.status_code == 200:
            xml_content = response.content
            root = ET.fromstring(xml_content)
            files = []
            for contents in root.findall('{http://s3.amazonaws.com/doc/2006-03-01/}Contents'):
                key = contents.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key').text
                files.append(key)
            return files
        else:
            raise Exception(f"Failed to access bucket. Status code: {response.status_code}")
    except Exception as e:
        raise Exception(f"Error listing files: {str(e)}")


def download_satellite_file(satellite_name, year, day_of_year, hour):
    """Download satellite file for specified parameters"""
    sat_info = SATELLITES[satellite_name]
    bucket = sat_info['bucket']
    identifier = sat_info['identifier']

    # Check date validity
    requested_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour)
    if not (sat_info['start_date'] <= requested_date <= sat_info['end_date']):
        raise Exception(f"Date {requested_date.strftime('%Y-%m-%d %H:00')} is outside {satellite_name} coverage period")

    bucket_url = f"https://{bucket}.s3.amazonaws.com/?prefix=ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/"
    files = list_files(bucket_url)

    # Determine file pattern based on satellite and date
    if satellite_name == 'GOES-16':
        transition_date = datetime(2019, 4, 2, 16)
        if requested_date < transition_date:
            file_pattern = f"OR_ABI-L1b-RadF-M3C13_{identifier}_s{year}{day_of_year:03d}{hour:02d}00"
        else:
            file_pattern = f"OR_ABI-L1b-RadF-M6C13_{identifier}_s{year}{day_of_year:03d}{hour:02d}00"
    else:
        file_pattern = f"OR_ABI-L1b-RadF-M6C13_{identifier}_s{year}{day_of_year:03d}{hour:02d}00"

    # Find matching file
    filename = ""
    for file in files:
        if file.startswith(f"ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/{file_pattern}"):
            filename = file.split('/')[-1]
            break

    if not filename:
        raise Exception(f"No {satellite_name} file found for the specified time")

    # Download file
    download_url = f"https://{bucket}.s3.amazonaws.com/ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/{filename}"

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
        save_path = tmp_file.name

    try:
        response = requests.get(download_url, timeout=60)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return save_path, filename
        else:
            raise Exception(f"File download failed. Status code: {response.status_code}")
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise Exception(f"Download error: {str(e)}")


def compute_lat_lon(ds):
    """Compute latitude and longitude from GOES projection"""
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]

    proj_info = ds.variables['goes_imager_projection']
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis

    X, Y = np.meshgrid(x, y)

    lambda_0 = np.deg2rad(lon_origin)
    a_var = np.sin(X) ** 2 + (np.cos(X) ** 2 * (np.cos(Y) ** 2 + ((r_eq ** 2 / r_pol ** 2) * np.sin(Y) ** 2)))
    b_var = -2 * H * np.cos(X) * np.cos(Y)
    c_var = H ** 2 - r_eq ** 2

    discriminant = b_var ** 2 - 4 * a_var * c_var
    discriminant = np.maximum(discriminant, 0)

    r_s = (-b_var - np.sqrt(discriminant)) / (2 * a_var)
    s_x = r_s * np.cos(X) * np.cos(Y)
    s_y = -r_s * np.sin(X)
    s_z = r_s * np.cos(X) * np.sin(Y)

    denominator = np.sqrt((H - s_x) ** 2 + s_y ** 2)

    if np.any(denominator == 0):
        raise ValueError("Zero values detected in coordinate computation")

    lat = np.rad2deg(np.arctan((r_eq ** 2 / r_pol ** 2) * (s_z / denominator)))
    lon = np.rad2deg(lambda_0 - np.arctan(s_y / (H - s_x)))

    return lat, lon


def rad_to_temp(radiance, planck_fk1, planck_fk2, planck_bc1, planck_bc2):
    """Convert radiance to brightness temperature"""
    brightness_temp = (planck_fk2 / (np.log((planck_fk1 / radiance) + 1)) - planck_bc1) / planck_bc2
    return brightness_temp - 273.15  # Convert from Kelvin to Celsius


def plot_goes_data(file_path, filename, center_coord):
    """Plot GOES satellite data"""
    with nc.Dataset(file_path) as ds:
        # Read radiance data
        data = ds.variables['Rad'][:].squeeze()
        lats, lons = compute_lat_lon(ds)

        # Get calibration coefficients
        planck_fk1 = ds.variables['planck_fk1'][0]
        planck_fk2 = ds.variables['planck_fk2'][0]
        planck_bc1 = ds.variables['planck_bc1'][0]
        planck_bc2 = ds.variables['planck_bc2'][0]

        # Convert to temperature
        data_temp_celsius = rad_to_temp(data, planck_fk1, planck_fk2, planck_bc1, planck_bc2)

        # Define region of interest
        center_lon, center_lat = center_coord
        lon_min, lon_max = center_lon - 10, center_lon + 10
        lat_min, lat_max = center_lat - 10, center_lat + 10

        # Ensure lons and lats are 2D
        if lons.ndim == 1:
            lons, _ = np.meshgrid(lons, lats)
        if lats.ndim == 1:
            _, lats = np.meshgrid(lons, lats)

        # Create masks for region of interest
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        mask = lon_mask & lat_mask

        if not np.any(mask):
            raise Exception("The requested location is not covered by this satellite!")

        # Apply masks
        data_box = np.ma.masked_where(~mask, data_temp_celsius)
        lons_box = np.ma.masked_where(~mask, lons)
        lats_box = np.ma.masked_where(~mask, lats)

        # Create plot
        custom_cmap = rbtop3()
        fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree()})

        vmin, vmax = -100, 40
        im = ax.pcolormesh(lons_box, lats_box, data_box, cmap=custom_cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Add map features
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linestyle=':')

        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='vertical', label='Temperature (°C)')

        # Extract date/time from filename and create title
        date_time_str = filename.split('_')[3][1:12]
        dt = datetime.strptime(date_time_str, '%Y%j%H%M')

        # Determine satellite from filename
        if 'G16' in filename:
            sat_name = 'GOES-16'
        elif 'G17' in filename:
            sat_name = 'GOES-17'
        elif 'G18' in filename:
            sat_name = 'GOES-18'
        elif 'G19' in filename:
            sat_name = 'GOES-19'
        else:
            sat_name = 'GOES'

        title = dt.strftime(f'{sat_name} Infrared Data for %B %d, %Y at %H:%M UTC')
        plt.title(title, fontsize=18, weight='bold', pad=10)

        fig.text(0.5, 0.085, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=15, weight='bold')

        return fig


def process_goes_data(date_input, hour, center_coord):
    """Main processing function for GOES data"""
    year = date_input.year
    day_of_year = date_input.timetuple().tm_yday

    requested_date = datetime(year, date_input.month, date_input.day, hour)
    center_lon, center_lat = center_coord

    # Select best satellite
    satellite_name, error = select_best_satellite(center_lon, center_lat, requested_date)
    if error:
        raise Exception(error)

    # Download and process data
    file_path, filename = download_satellite_file(satellite_name, year, day_of_year, hour)

    try:
        fig = plot_goes_data(file_path, filename, center_coord)
        return fig, satellite_name
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# Streamlit UI
st.title("🛰️ GOES Satellite Data Viewer")
st.markdown("### High-Resolution Infrared Satellite Imagery")

st.markdown("""
This tool provides access to **GOES-16, GOES-17, GOES-18, and GOES-19** satellite data.
The system automatically selects the best satellite based on your location and date.
""")

# Create columns for layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📅 Select Date & Time")

    # Date input
    today = datetime.now().date()
    min_date = datetime(2017, 4, 1).date()

    date_input = st.date_input(
        "Date",
        value=today,
        min_value=min_date,
        max_value=today
    )

    # Hour selection
    hour_input = st.selectbox(
        "Hour (UTC)",
        options=list(range(24)),
        index=12,
        format_func=lambda x: f"{x:02d}:00"
    )

    st.subheader("🌍 Location")

    # Location input method
    location_method = st.radio(
        "How would you like to specify the location?",
        ["City/Place Name", "Coordinates (Lat, Lon)"]
    )

    if location_method == "City/Place Name":
        location_input = st.text_input(
            "Enter city or place name",
            placeholder="e.g., Miami, Los Angeles, Honolulu"
        )
        lat, lon = None, None
        if location_input:
            with st.spinner("Geocoding location..."):
                lat, lon = get_coordinates(location_input)
                if lat and lon:
                    st.success(f"📍 Found: {lat:.4f}°, {lon:.4f}°")
                else:
                    st.error("Location not found. Please try a different name or use coordinates.")
    else:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=25.7617, step=0.1)
        with col_lon:
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-80.1918, step=0.1)

    # Satellite selection preview
    if lat is not None and lon is not None:
        try:
            requested_date = datetime(date_input.year, date_input.month, date_input.day, hour_input)
            selected_sat, error = select_best_satellite(lon, lat, requested_date)
            if selected_sat:
                sat_info = SATELLITES[selected_sat]
                st.info(f"🛰️ Will use: **{selected_sat}** (positioned at {sat_info['longitude']}°W)")
            else:
                st.warning(f"⚠️ {error}")
        except:
            pass

    # Generate button
    generate_button = st.button("🚀 Generate Satellite Image", type="primary")

with col2:
    st.subheader("📊 Satellite Image")

    if generate_button:
        if lat is not None and lon is not None:
            try:
                with st.spinner("Downloading and processing GOES satellite data... This may take a moment."):
                    fig, satellite_used = process_goes_data(date_input, hour_input, (lon, lat))

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)  # Clean up memory

                st.success(f"✅ Image generated successfully using {satellite_used}!")
                st.info("💡 Right-click on the image to save it to your device.")

            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")
        else:
            st.warning("⚠️ Please provide a valid location.")

# Information section
with st.expander("ℹ️ About GOES Satellites"):
    st.markdown("""
    **GOES (Geostationary Operational Environmental Satellite)** provides continuous monitoring of weather and environmental conditions.

    **Satellite Coverage:**
    - **GOES-16** (75.2°W): April 2017 - October 2024 | Eastern US, Atlantic, South America
    - **GOES-17** (137.0°W): August 2018 - July 2022 | Western US, Pacific
    - **GOES-18** (137.0°W): July 2022 - Present | Western US, Pacific  
    - **GOES-19** (75.2°W): October 2024 - Present | Eastern US, Atlantic, South America

    **Features:**
    - **Temporal Resolution**: Hourly data available
    - **Spatial Resolution**: ~2km at satellite nadir point
    - **Channel**: Infrared (10.3 μm) - shows cloud-top and surface temperatures
    - **Data Source**: NOAA/NESDIS via AWS Open Data

    The system automatically selects the best satellite based on:
    1. **Geographic proximity** to your location
    2. **Temporal availability** for your selected date

    Colder temperatures (blues/purples) indicate higher cloud tops or cold surfaces,
    while warmer temperatures (reds/yellows) show lower clouds or warm surfaces.
    """)

# Coverage map
with st.expander("🗺️ Satellite Coverage Areas"):
    st.markdown("""
    **GOES-16 & GOES-19** (75.2°W) cover:
    - Eastern United States and Canada
    - Caribbean and Central America  
    - Northern and Eastern South America
    - Western Atlantic Ocean

    **GOES-17 & GOES-18** (137.0°W) cover:
    - Western United States and Canada
    - Mexico and Central America
    - Western South America  
    - Eastern Pacific Ocean
    - Hawaii and Alaska
    """)

st.markdown("---")
st.markdown("*Created by Sekai Chandra (@Sekai_WX)*")