#!/bin/bash

# Data folder link management script
# Usage: ./data_link_manager.sh --link <DATA_FOLDER> or ./data_link_manager.sh --unlink

# List of folders to link
FOLDERS=("data_normal" "data_more" "data_test" "data_true")

# Help function
show_help() {
    echo "Usage:"
    echo "  $0 --link <DATA_FOLDER>    Create links to subfolders of specified DATA_FOLDER in current directory"
    echo "  $0 --unlink                Remove data links from current directory"
    echo ""
    echo "Folders to be linked: ${FOLDERS[*]}"
}

# Link creation function
create_links() {
    local data_folder="$1"
    
    # Convert to absolute path
    data_folder=$(realpath "$data_folder")
    
    # Check if DATA_FOLDER exists
    if [[ ! -d "$data_folder" ]]; then
        echo "Error: Directory '$data_folder' does not exist."
        exit 1
    fi
    
    echo "Data folder: $data_folder"
    echo "Creating links in current directory..."
    
    for folder in "${FOLDERS[@]}"; do
        source_path="$data_folder/$folder"
        target_path="$PWD/$folder"
        
        # Create source folder if it doesn't exist
        if [[ ! -d "$source_path" ]]; then
            echo "Folder '$source_path' does not exist, creating it."
            mkdir -p "$source_path"
        fi
        
        # Check if file or folder already exists at target location
        if [[ -e "$target_path" ]]; then
            if [[ -L "$target_path" ]]; then
                echo "Warning: Symbolic link already exists at '$target_path'. Removing and recreating."
                unlink "$target_path"
            else
                echo "Error: File or folder already exists at '$target_path'. Please remove manually and try again."
                continue
            fi
        fi
        
        # Create symbolic link
        ln -s "$source_path" "$target_path"
        if [[ $? -eq 0 ]]; then
            echo "✓ $folder link created successfully: $target_path -> $source_path"
        else
            echo "✗ $folder link creation failed"
        fi
    done
    
    echo "Link creation process completed."
}

# Link removal function
remove_links() {
    echo "Removing data links from current directory..."
    
    for folder in "${FOLDERS[@]}"; do
        target_path="$PWD/$folder"
        
        if [[ -L "$target_path" ]]; then
            unlink "$target_path"
            if [[ $? -eq 0 ]]; then
                echo "✓ $folder link removed successfully: $target_path"
            else
                echo "✗ $folder link removal failed: $target_path"
            fi
        elif [[ -e "$target_path" ]]; then
            echo "Warning: '$target_path' is not a symbolic link. Leaving it untouched."
        else
            echo "- $folder link does not exist: $target_path"
        fi
    done
    
    echo "Link removal process completed."
}

# Main logic
case "$1" in
    --link)
        if [[ -z "$2" ]]; then
            echo "Error: --link option requires DATA_FOLDER path."
            show_help
            exit 1
        fi
        create_links "$2"
        ;;
    --unlink)
        remove_links
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo "Error: Invalid option."
        show_help
        exit 1
        ;;
esac
