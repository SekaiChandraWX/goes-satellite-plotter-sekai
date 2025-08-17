import streamlit as st
import os
import requests
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
import gc  # Garbage collection
from threading import Lock

# Set page config
st.set_page_config(
    page_title="GOES Satellite Viewer", 
    page_icon="🛰️",
    layout="wide"
)

# Resource management
processing_lock = Lock()
MAX_MEMORY_USAGE = 500  # MB limit

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
    """Lightweight colormap for temperature visualization"""
    colors = [
        "#000000", "#fffdfd", "#05fcfe", "#010071", "#00fe24", 
        "#fbff2d", "#fd1917", "#000300", "#e1e4e5", "#eb6fc0", 
        "#9b1f94", "#330f2f"
    ]
    positions = [0, 60/140, 60/140, 70/140, 80/140, 90/140, 
                100/140, 110/140, 120/140, 120/140, 130/140, 1.0]
    
    newcmp = mcolors.LinearSegmentedColormap.from_list("", 
        list(zip(positions, colors)))
    return newcmp.reversed()

def get_coordinates(location_str):
    """Lightweight geocoding with timeout and rate limiting"""
    try:
        geolocator = Nominatim(user_agent="goes_satellite_viewer", timeout=5)
        time.sleep(2)  # Rate limiting
        location = geolocator.geocode(location_str, timeout=5)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return None, None

def select_best_satellite(lon, lat, requested_date):
    """Lightweight satellite selection"""
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
    """List files with retry logic and timeouts"""
    for attempt in range(max_retries):
        try:
            time.sleep(attempt * 2)  # Progressive delay
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

def download_satellite_file_chunked(satellite_name, year, day_of_year, hour, progress_bar=None):
    """Memory-efficient chunked download with progress tracking"""
    sat_info = SATELLITES[satellite_name]
    bucket = sat_info['bucket']
    identifier = sat_info['identifier']
    
    # Validate date
    requested_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour)
    if not (sat_info['start_date'] <= requested_date <= sat_info['end_date']):
        raise Exception(f"Date outside {satellite_name} coverage period")
    
    if progress_bar:
        progress_bar.progress(10, "Searching for satellite files...")
    
    bucket_url = f"https://{bucket}.s3.amazonaws.com/?prefix=ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/"
    files = list_files_with_retry(bucket_url)
    
    # Determine file pattern
    if satellite_name == 'GOES-16':
        transition_date = datetime(2019, 4, 2, 16)
        mode = "M3" if requested_date < transition_date else "M6"
    else:
        mode = "M6"
    
    file_pattern = f"OR_ABI-L1b-RadF-{mode}C13_{identifier}_s{year}{day_of_year:03d}{hour:02d}00"
    
    # Find file
    filename = ""
    for file in files:
        if file.startswith(f"ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/{file_pattern}"):
            filename = file.split('/')[-1]
            break
    
    if not filename:
        raise Exception(f"No {satellite_name} file found for the specified time")
    
    if progress_bar:
        progress_bar.progress(25, "Downloading satellite data...")
    
    # Download with chunked streaming
    download_url = f"https://{bucket}.s3.amazonaws.com/ABI-L1b-RadF/{year}/{day_of_year:03d}/{hour:02d}/{filename}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
        save_path = tmp_file.name
    
    try:
        response = requests.get(download_url, timeout=120, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):  # Small chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_bar and total_size > 0:
                            progress = 25 + int(50 * downloaded / total_size)
                            progress_bar.progress(progress, f"Downloading: {downloaded/(1024*1024):.1f}MB")
                        time.sleep(0.01)  # Prevent CPU overload
            
            return save_path, filename
        else:
            raise Exception(f"Download failed. Status: {response.status_code}")
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise Exception(f"Download error: {str(e)}")

def compute_lat_lon(ds):
    """Compute latitude and longitude from GOES projection - using original working method"""
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

def plot_goes_data_optimized(file_path, filename, center_coord, progress_bar=None):
    """Optimized plotting using original working method for coverage detection"""
    
    if progress_bar:
        progress_bar.progress(80, "Processing satellite data...")
    
    def rad_to_temp(radiance, planck_fk1, planck_fk2, planck_bc1, planck_bc2):
        """Convert radiance to brightness temperature - original method"""
        brightness_temp = (planck_fk2 / (np.log((planck_fk1 / radiance) + 1)) - planck_bc1) / planck_bc2
        return brightness_temp - 273.15  # Convert from Kelvin to Celsius
    
    with nc.Dataset(file_path) as ds:
        # Use original method: load full data first
        data = ds.variables['Rad'][:].squeeze()
        lats, lons = compute_lat_lon(ds)  # Full coordinate grid
        
        time.sleep(0.2)  # Brief pause for CPU
        
        # Get calibration coefficients
        planck_fk1 = ds.variables['planck_fk1'][0]
        planck_fk2 = ds.variables['planck_fk2'][0]
        planck_bc1 = ds.variables['planck_bc1'][0]
        planck_bc2 = ds.variables['planck_bc2'][0]

        # Convert to temperature using original method
        data_temp_celsius = rad_to_temp(data, planck_fk1, planck_fk2, planck_bc1, planck_bc2)
        
        # Clear original data to save memory
        del data
        gc.collect()
        
        if progress_bar:
            progress_bar.progress(85, "Checking location coverage...")
        
        # Use original coverage detection method
        center_lon, center_lat = center_coord
        lon_min, lon_max = center_lon - 10, center_lon + 10
        lat_min, lat_max = center_lat - 10, center_lat + 10

        # Ensure lons and lats are 2D - original method
        if lons.ndim == 1:
            lons, _ = np.meshgrid(lons, lats)
        if lats.ndim == 1:
            _, lats = np.meshgrid(lons, lats)

        # Original masking logic
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        mask = lon_mask & lat_mask

        # Original coverage check
        if not np.any(mask):
            raise Exception("This location is not covered by this satellite! Please check pinned messages for location specifics!")

        # Original masking application
        data_box = np.ma.masked_where(~mask, data_temp_celsius)
        lons_box = np.ma.masked_where(~mask, lons)
        lats_box = np.ma.masked_where(~mask, lats)
        
        # Clear full arrays to save memory
        del data_temp_celsius, lons, lats, mask, lon_mask, lat_mask
        gc.collect()
        
        if progress_bar:
            progress_bar.progress(90, "Creating visualization...")
        
        time.sleep(0.3)  # CPU break

        # NOW apply subsampling to the extracted region for plotting optimization
        subsample = 3  # Reduced subsampling for better quality
        data_plot = data_box[::subsample, ::subsample]
        lons_plot = lons_box[::subsample, ::subsample]
        lats_plot = lats_box[::subsample, ::subsample]
        
        # Clear intermediate arrays
        del data_box, lons_box, lats_box
        gc.collect()

        # Create plot - using original settings but optimized figure size
        custom_cmap = rbtop3()
        fig, ax = plt.subplots(figsize=(16, 10), dpi=150, 
                              subplot_kw={'projection': ccrs.PlateCarree()})

        vmin, vmax = -100, 40
        
        # Original plotting method
        im = ax.pcolormesh(lons_plot, lats_plot, data_plot, cmap=custom_cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), shading='auto')

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Add map features - original method
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linestyle=':')

        # Original colorbar
        plt.colorbar(im, ax=ax, orientation='vertical', label='Temperature (°C)')

        # Original title generation
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

        # Original attribution
        fig.text(0.5, 0.085, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=15, weight='bold')

        # Clean up memory
        del data_plot, lons_plot, lats_plot
        gc.collect()

        return fig

def process_goes_data_optimized(date_input, hour, center_coord):
    """Main processing with resource management"""
    
    # Resource lock to prevent concurrent processing
    if not processing_lock.acquire(blocking=False):
        raise Exception("Another satellite request is processing. Please wait and try again.")
    
    try:
        year = date_input.year
        day_of_year = date_input.timetuple().tm_yday
        
        requested_date = datetime(year, date_input.month, date_input.day, hour)
        center_lon, center_lat = center_coord
        
        # Progress tracking
        progress_bar = st.progress(0, "Initializing satellite data request...")
        
        # Select satellite
        satellite_name, error = select_best_satellite(center_lon, center_lat, requested_date)
        if error:
            raise Exception(error)
        
        progress_bar.progress(5, f"Selected {satellite_name} for optimal coverage...")
        time.sleep(1)
        
        # Download data
        file_path, filename = download_satellite_file_chunked(
            satellite_name, year, day_of_year, hour, progress_bar)
        
        try:
            # Process and plot
            fig = plot_goes_data_optimized(file_path, filename, center_coord, progress_bar)
            progress_bar.progress(100, "Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            
            return fig, satellite_name
            
        finally:
            # Always cleanup file
            if os.path.exists(file_path):
                os.remove(file_path)
            gc.collect()
            
    finally:
        processing_lock.release()

# Streamlit UI
st.title("🛰️ GOES Satellite Data Viewer")
st.markdown("### High-Resolution Infrared Satellite Imagery")

st.info("⚡ **Optimized for Streamlit Cloud**: Processing may take 1-3 minutes for high-quality results")

st.markdown("""
Access **GOES-16, GOES-17, GOES-18, and GOES-19** satellite data with automatic satellite selection
based on your location and date. Optimized for reliable cloud processing.
""")

# Layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📅 Date & Time")
    
    today = datetime.now().date()
    min_date = datetime(2017, 4, 1).date()
    
    date_input = st.date_input(
        "Date", value=today, min_value=min_date, max_value=today)
    
    hour_input = st.selectbox(
        "Hour (UTC)", options=list(range(0, 24, 2)), index=6,  # Every 2 hours
        format_func=lambda x: f"{x:02d}:00")
    
    st.subheader("🌍 Location")
    
    location_method = st.radio(
        "Location Input", ["City/Place Name", "Coordinates"])
    
    if location_method == "City/Place Name":
        location_input = st.text_input(
            "Enter location", placeholder="Miami, Los Angeles, Denver")
        lat, lon = None, None
        if location_input:
            with st.spinner("Finding coordinates..."):
                lat, lon = get_coordinates(location_input)
                if lat and lon:
                    st.success(f"📍 {lat:.3f}°, {lon:.3f}°")
                else:
                    st.error("Location not found")
    else:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude", min_value=-60.0, max_value=60.0, 
                                value=25.8, step=0.1)
        with col_lon:
            lon = st.number_input("Longitude", min_value=-160.0, max_value=-40.0, 
                                value=-80.2, step=0.1)
    
    # Satellite preview
    if lat is not None and lon is not None:
        try:
            req_date = datetime(date_input.year, date_input.month, date_input.day, hour_input)
            selected_sat, _ = select_best_satellite(lon, lat, req_date)
            if selected_sat:
                sat_info = SATELLITES[selected_sat]
                st.info(f"🛰️ **{selected_sat}** @ {sat_info['longitude']}°W")
        except:
            pass
    
    generate_button = st.button("🚀 Generate Image", type="primary")

with col2:
    st.subheader("📊 Satellite Image")
    
    if generate_button:
        if lat is not None and lon is not None:
            try:
                fig, satellite_used = process_goes_data_optimized(date_input, hour_input, (lon, lat))
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                gc.collect()
                
                st.success(f"✅ Generated using {satellite_used}")
                st.caption("💡 Right-click to save image")
                
            except Exception as e:
                st.error(f"❌ {str(e)}")
                st.info("💡 Try a different time or location if the error persists")
        else:
            st.warning("⚠️ Please provide a valid location")

# Information sections
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    **Optimized GOES Satellite Data Viewer**
    
    **Performance Features:**
    - **Smart subsampling**: Reduces data size while maintaining quality
    - **Chunked downloads**: Streams large files efficiently  
    - **Memory management**: Automatic cleanup prevents crashes
    - **Progress tracking**: Real-time status updates
    
    **Satellite Coverage:**
    - **GOES-16/19** (75°W): Eastern Americas, Atlantic
    - **GOES-17/18** (137°W): Western Americas, Pacific
    
    **Data Quality:**
    - **Resolution**: ~8km effective (optimized from ~2km native)
    - **Update**: Every 2 hours for reliability
    - **Coverage**: 2017 to present
    """)

with st.expander("⚡ Performance Notes"):
    st.markdown("""
    **Why the wait time?**
    - GOES files are 50-100MB each
    - Complex coordinate transformations required
    - High-resolution plotting with geographic projections
    
    **Optimizations Applied:**
    - Data subsampling (4x reduction)
    - Memory-efficient processing
    - Progressive downloads with chunking
    - Automatic resource cleanup
    
    **Best Practices:**
    - Allow 1-3 minutes for processing
    - Use recent dates for faster access
    - Avoid concurrent requests
    """)

st.markdown("---")
st.markdown("*Optimized for Streamlit Cloud by Sekai Chandra (@Sekai_WX)*")