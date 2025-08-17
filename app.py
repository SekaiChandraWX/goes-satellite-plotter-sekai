import streamlit as st
import os
import requests
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
import gc
from threading import Lock

# Set page config
st.set_page_config(
    page_title="GOES Satellite Viewer", 
    page_icon="🛰️",
    layout="wide"
)

# Resource management
processing_lock = Lock()

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
        'end_date': datetime(2030, 12, 31, 23, 59),
        'bucket': 'noaa-goes18',
        'identifier': 'G18'
    },
    'GOES-19': {
        'longitude': -75.2,
        'start_date': datetime(2024, 10, 10, 20, 0),
        'end_date': datetime(2030, 12, 31, 23, 59),
        'bucket': 'noaa-goes19',
        'identifier': 'G19'
    }
}

def rbtop3():
    """Original rbtop3 colormap - EXACTLY as provided"""
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
    """Geocoding with timeout"""
    try:
        geolocator = Nominatim(user_agent="goes_satellite_viewer", timeout=5)
        time.sleep(1)
        location = geolocator.geocode(location_str, timeout=5)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return None, None

def select_best_satellite(lon, lat, requested_date):
    """Select optimal satellite"""
    distances = {}
    available_satellites = []
    
    for sat_name, sat_info in SATELLITES.items():
        if sat_info['start_date'] <= requested_date <= sat_info['end_date']:
            available_satellites.append(sat_name)
            distance = abs(lon - sat_info['longitude'])
            if distance > 180:
                distance = 360 - distance
            distances[sat_name] = distance
    
    if not available_satellites:
        return None, "No satellite data available for the requested date"
    
    best_satellite = min(available_satellites, key=lambda x: distances[x])
    return best_satellite, None

def list_files_with_retry(bucket_url, max_retries=3):
    """File listing with retry"""
    for attempt in range(max_retries):
        try:
            time.sleep(attempt * 1)
            response = requests.get(bucket_url, timeout=15)
            if response.status_code == 200:
                xml_content = response.content
                root = ET.fromstring(xml_content)
                files = []
                for contents in root.findall('{http://s3.amazonaws.com/doc/2006-03-01/}Contents'):
                    key = contents.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key').text
                    files.append(key)
                return files
            else:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to access bucket after {max_retries} attempts")
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Error listing files: {str(e)}")
            time.sleep(2)
    return []

def download_satellite_file(satellite_name, year, day_of_year, hour, progress_bar):
    """Standard download with progress"""
    sat_info = SATELLITES[satellite_name]
    bucket = sat_info['bucket']
    identifier = sat_info['identifier']
    
    requested_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour)
    if not (sat_info['start_date'] <= requested_date <= sat_info['end_date']):
        raise Exception(f"Date outside {satellite_name} coverage period")
    
    progress_bar.progress(10, "Searching for satellite files...")
    
    bucket_url = f"https://{bucket}.s3.amazonaws.com/?prefix=ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/"
    files = list_files_with_retry(bucket_url)
    
    # File pattern logic
    if satellite_name == 'GOES-16':
        transition_date = datetime(2019, 4, 2, 16)
        mode = "M3" if requested_date < transition_date else "M6"
    else:
        mode = "M6"
    
    file_pattern = f"OR_ABI-L1b-RadF-{mode}C13_{identifier}_s{year}{day_of_year:03d}{hour:02d}00"
    
    filename = ""
    for file in files:
        if file.startswith(f"ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/{file_pattern}"):
            filename = file.split('/')[-1]
            break
    
    if not filename:
        raise Exception(f"No {satellite_name} file found for the specified time")
    
    progress_bar.progress(25, "Downloading satellite data...")
    
    download_url = f"https://{bucket}.s3.amazonaws.com/ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/{filename}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
        save_path = tmp_file.name
    
    try:
        response = requests.get(download_url, timeout=120, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_bar and total_size > 0:
                            progress = 25 + int(50 * downloaded / total_size)
                            progress_bar.progress(progress, f"Downloading: {downloaded/(1024*1024):.1f}MB")
                        time.sleep(0.005)  # Small CPU break
            
            return save_path, filename
        else:
            raise Exception(f"Download failed. Status: {response.status_code}")
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise Exception(f"Download error: {str(e)}")

def find_region_indices(x_coords, y_coords, center_lon, center_lat, proj_info):
    """Find array indices for region of interest WITHOUT loading full data"""
    
    # Get projection parameters
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis
    
    # Define target region
    region_size = 10
    lon_min, lon_max = center_lon - region_size, center_lon + region_size
    lat_min, lat_max = center_lat - region_size, center_lat + region_size
    
    # Sample coordinates to find approximate indices
    # Use every 50th point to quickly map the coordinate space
    x_sample = x_coords[::50]
    y_sample = y_coords[::50]
    
    X_sample, Y_sample = np.meshgrid(x_sample, y_sample)
    
    # Quick coordinate transformation for sampling
    lambda_0 = np.deg2rad(lon_origin)
    a_var = np.sin(X_sample) ** 2 + (np.cos(X_sample) ** 2 * (np.cos(Y_sample) ** 2 + ((r_eq ** 2 / r_pol ** 2) * np.sin(Y_sample) ** 2)))
    b_var = -2 * H * np.cos(X_sample) * np.cos(Y_sample)
    c_var = H ** 2 - r_eq ** 2
    
    discriminant = np.maximum(b_var ** 2 - 4 * a_var * c_var, 0)
    r_s = (-b_var - np.sqrt(discriminant)) / (2 * a_var)
    s_x = r_s * np.cos(X_sample) * np.cos(Y_sample)
    s_y = -r_s * np.sin(X_sample)
    s_z = r_s * np.cos(X_sample) * np.sin(Y_sample)
    
    denominator = np.sqrt((H - s_x) ** 2 + s_y ** 2)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    lat_sample = np.rad2deg(np.arctan((r_eq ** 2 / r_pol ** 2) * (s_z / denominator)))
    lon_sample = np.rad2deg(lambda_0 - np.arctan(s_y / (H - s_x)))
    
    # Find bounds in sampled space
    region_mask = ((lon_sample >= lon_min) & (lon_sample <= lon_max) & 
                   (lat_sample >= lat_min) & (lat_sample <= lat_max))
    
    if not np.any(region_mask):
        raise Exception("This location is not covered by this satellite!")
    
    # Find index bounds with padding
    y_indices, x_indices = np.where(region_mask)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise Exception("Location not found in satellite coverage!")
    
    # Convert sampled indices back to full resolution indices
    y_min_idx = max(0, (y_indices.min() * 50) - 100)  # Add padding
    y_max_idx = min(len(y_coords), (y_indices.max() * 50) + 100)
    x_min_idx = max(0, (x_indices.min() * 50) - 100)
    x_max_idx = min(len(x_coords), (x_indices.max() * 50) + 100)
    
    return y_min_idx, y_max_idx, x_min_idx, x_max_idx

def process_goes_data_regional(file_path, filename, center_coord, progress_bar):
    """FULL RESOLUTION processing using regional data loading"""
    
    progress_bar.progress(80, "Processing satellite data...")
    
    center_lon, center_lat = center_coord
    
    with nc.Dataset(file_path, 'r') as ds:
        
        # Load coordinate arrays (these are small - 1D)
        x_coords = ds.variables['x'][:]
        y_coords = ds.variables['y'][:]
        proj_info = ds.variables['goes_imager_projection']
        
        progress_bar.progress(82, "Finding region indices...")
        
        # Find which part of the array we need
        y_min, y_max, x_min, x_max = find_region_indices(
            x_coords, y_coords, center_lon, center_lat, proj_info)
        
        progress_bar.progress(85, "Loading regional data...")
        
        # Load ONLY the region we need - this is the key memory saver!
        data_region = ds.variables['Rad'][y_min:y_max, x_min:x_max].squeeze()
        x_region = x_coords[x_min:x_max]
        y_region = y_coords[y_min:y_max]
        
        # Get calibration coefficients
        planck_fk1 = ds.variables['planck_fk1'][0]
        planck_fk2 = ds.variables['planck_fk2'][0]
        planck_bc1 = ds.variables['planck_bc1'][0]
        planck_bc2 = ds.variables['planck_bc2'][0]
        
        progress_bar.progress(88, "Computing coordinates...")
        
        # Compute coordinates for our region only
        X_region, Y_region = np.meshgrid(x_region, y_region)
        
        # Full coordinate computation for region
        lon_origin = proj_info.longitude_of_projection_origin
        H = proj_info.perspective_point_height + proj_info.semi_major_axis
        r_eq = proj_info.semi_major_axis
        r_pol = proj_info.semi_minor_axis
        
        lambda_0 = np.deg2rad(lon_origin)
        a_var = np.sin(X_region) ** 2 + (np.cos(X_region) ** 2 * (np.cos(Y_region) ** 2 + ((r_eq ** 2 / r_pol ** 2) * np.sin(Y_region) ** 2)))
        b_var = -2 * H * np.cos(X_region) * np.cos(Y_region)
        c_var = H ** 2 - r_eq ** 2
        
        discriminant = np.maximum(b_var ** 2 - 4 * a_var * c_var, 0)
        r_s = (-b_var - np.sqrt(discriminant)) / (2 * a_var)
        s_x = r_s * np.cos(X_region) * np.cos(Y_region)
        s_y = -r_s * np.sin(X_region)
        s_z = r_s * np.cos(X_region) * np.sin(Y_region)
        
        denominator = np.sqrt((H - s_x) ** 2 + s_y ** 2)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        lats_region = np.rad2deg(np.arctan((r_eq ** 2 / r_pol ** 2) * (s_z / denominator)))
        lons_region = np.rad2deg(lambda_0 - np.arctan(s_y / (H - s_x)))
        
        # Clean up coordinate computation arrays
        del X_region, Y_region, a_var, b_var, c_var, discriminant, r_s, s_x, s_y, s_z, denominator
        gc.collect()
        
        progress_bar.progress(92, "Converting to temperature...")
        
        # Convert radiance to temperature - FULL PRECISION
        def rad_to_temp(radiance, planck_fk1, planck_fk2, planck_bc1, planck_bc2):
            brightness_temp = (planck_fk2 / (np.log((planck_fk1 / radiance) + 1)) - planck_bc1) / planck_bc2
            return brightness_temp - 273.15
        
        data_temp_celsius = rad_to_temp(data_region, planck_fk1, planck_fk2, planck_bc1, planck_bc2)
        
        del data_region
        gc.collect()
        
        progress_bar.progress(95, "Creating final plot...")
        
        # Extract the actual display region
        region_size = 10
        lon_min, lon_max = center_lon - region_size, center_lon + region_size
        lat_min, lat_max = center_lat - region_size, center_lat + region_size
        
        # Mask for final display region
        display_mask = ((lons_region >= lon_min) & (lons_region <= lon_max) & 
                       (lats_region >= lat_min) & (lats_region <= lat_max))
        
        if not np.any(display_mask):
            raise Exception("This location is not covered by this satellite!")
        
        # Apply display mask
        data_display = np.ma.masked_where(~display_mask, data_temp_celsius)
        lons_display = np.ma.masked_where(~display_mask, lons_region)
        lats_display = np.ma.masked_where(~display_mask, lats_region)
        
        del data_temp_celsius, lons_region, lats_region, display_mask
        gc.collect()
        
        # Create plot with ORIGINAL quality and colormap
        custom_cmap = rbtop3()  # ORIGINAL colormap
        
        fig, ax = plt.subplots(figsize=(18, 10), dpi=300,  # ORIGINAL high quality
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        vmin, vmax = -100, 40  # ORIGINAL scale
        
        im = ax.pcolormesh(lons_display, lats_display, data_display, 
                          cmap=custom_cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), shading='auto')  # ORIGINAL shading
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        
        # ORIGINAL map features
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linestyle=':')
        
        # ORIGINAL colorbar
        plt.colorbar(im, ax=ax, orientation='vertical', label='Temperature (°C)')
        
        # ORIGINAL title format
        date_time_str = filename.split('_')[3][1:12]
        dt = datetime.strptime(date_time_str, '%Y%j%H%M')
        
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
        
        # ORIGINAL attribution
        fig.text(0.5, 0.085, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=15, weight='bold')
        
        del data_display, lons_display, lats_display
        gc.collect()
        
        return fig

def process_goes_data_optimized(date_input, hour, center_coord):
    """Main processing with regional loading for memory efficiency"""
    
    if not processing_lock.acquire(blocking=False):
        raise Exception("Another satellite request is processing. Please wait and try again.")
    
    try:
        year = date_input.year
        day_of_year = date_input.timetuple().tm_yday
        
        requested_date = datetime(year, date_input.month, date_input.day, hour)
        center_lon, center_lat = center_coord
        
        progress_bar = st.progress(0, "Initializing satellite data request...")
        
        # Select satellite
        satellite_name, error = select_best_satellite(center_lon, center_lat, requested_date)
        if error:
            raise Exception(error)
        
        progress_bar.progress(5, f"Selected {satellite_name} for optimal coverage...")
        
        # Download data
        file_path, filename = download_satellite_file(satellite_name, year, day_of_year, hour, progress_bar)
        
        try:
            # Process with regional loading
            fig = process_goes_data_regional(file_path, filename, center_coord, progress_bar)
            progress_bar.progress(100, "Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            
            return fig, satellite_name
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            gc.collect()
            
    finally:
        processing_lock.release()

# Streamlit UI
st.title("🛰️ GOES Satellite Data Viewer")
st.markdown("### High-Resolution Infrared Satellite Imagery")

st.markdown("""
Access **GOES-16, GOES-17, GOES-18, and GOES-19** satellite data with automatic satellite selection
based on your location and date. Generate professional-quality infrared imagery from the latest 
geostationary weather satellites.
""")

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📅 Select Date & Time")
    
    today = datetime.now().date()
    min_date = datetime(2017, 4, 1).date()
    
    date_input = st.date_input(
        "Date", value=today, min_value=min_date, max_value=today)
    
    hour_input = st.selectbox(
        "Hour (UTC)", options=list(range(24)), index=12,
        format_func=lambda x: f"{x:02d}:00")
    
    st.subheader("🌍 Location")
    
    location_method = st.radio(
        "How would you like to specify the location?",
        ["City/Place Name", "Coordinates (Lat, Lon)"])
    
    if location_method == "City/Place Name":
        location_input = st.text_input(
            "Enter city or place name",
            placeholder="e.g., Miami, Los Angeles, Honolulu")
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
    
    generate_button = st.button("🚀 Generate Satellite Image", type="primary")

with col2:
    st.subheader("📊 Satellite Image")
    
    if generate_button:
        if lat is not None and lon is not None:
            try:
                with st.spinner("Processing GOES satellite data... This may take 1-3 minutes."):
                    fig, satellite_used = process_goes_data_optimized(date_input, hour_input, (lon, lat))
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                gc.collect()
                
                st.success(f"✅ Image generated successfully using {satellite_used}!")
                st.info("💡 Right-click on the image to save it to your device.")
                
            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")
        else:
            st.warning("⚠️ Please provide a valid location.")

# Information section
with st.expander("🛰️ About GOES Satellites"):
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