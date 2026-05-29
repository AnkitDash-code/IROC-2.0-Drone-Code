import os

def update_events(status_message):
    file_path = "/home/jetson123/Desktop/safe_spots/events.txt"
    
    try:
        with open(file_path, 'a') as f:
            f.write(status_message + '\n')
        print(f"Events updated: {status_message}")
    except Exception as e:
        print(f"Error updating events: {e}")

def update_safe_spot(x, y, spot_number=None):
    file_path = "/home/jetson123/Desktop/safe_spots/safe_spots.txt"
    
    try:
        # Read existing content
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
        else:
            # Create default file structure with Arena and Safe Spots sections
            lines = [
                "Arena:\n",
                "Corner1: [37.7749, -122.4194]\n",
                "Corner2: [37.7749, -122.4144]\n", 
                "Corner3: [37.7699, -122.4144]\n",
                "Corner4: [37.7699, -122.4194]\n",
                "\n",
                "Detected Safe Spots\n",
                "SafeSpots:\n"
            ]
        
        # Find or create the spot line
        spot_line = f"Spot{spot_number if spot_number else 1}: [{x}, {y}]\n"
        
        # If spot_number is specified, try to update existing spot
        if spot_number:
            spot_found = False
            for i, line in enumerate(lines):
                if f"Spot{spot_number}:" in line:
                    lines[i] = spot_line
                    spot_found = True
                    break
            
            # If spot not found, add it
            if not spot_found:
                lines.append(spot_line)
        else:
            # Find next available spot number
            existing_spots = []
            for line in lines:
                if "Spot" in line and ":" in line:
                    try:
                        spot_num = int(line.split("Spot")[1].split(":")[0])
                        existing_spots.append(spot_num)
                    except:
                        pass
            
            next_spot = max(existing_spots) + 1 if existing_spots else 1
            spot_line = f"Spot{next_spot}: [{x}, {y}]\n"
            lines.append(spot_line)
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f"Safe spot updated: Spot{spot_number if spot_number else 'new'} = [{x}, {y}]")
        
    except Exception as e:
        print(f"Error updating safe spot: {e}")

def main():
    for i in range(4):
        print(i)
        update_safe_spot(x=i,y=i+1)

if __name__ == "__main__":
    main()