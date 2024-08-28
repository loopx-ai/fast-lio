#ifndef _NAME_PCD_H_
#define _NAME_PCD_H_

#include <time.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>

#define INVALID_PATH 0
#define ISFOLDER     1
#define ISFILE       2

#define DEFAULTNAME  0
#define TIMENAME     1
#define CUSTOMNAME   2

struct FILE_PATH {
    std::string filepath;
    std::string filename;
    std::string prefix;
    std::string extension;
};

std::string ExtractDirectory( const std::string& path )
{
    return path.substr( 0, path.find_last_of("/\\") +1 );
}

std::string ExtractFilename( const std::string& path )
{

    int lastdash = path.find_last_of("/\\");
    int lastdot = path.find_last_of(".");
    std::cout << "lastdash = " << lastdash <<", lastdot = "<< lastdot<<std::endl;

    if (path.find_last_of(".") == -1){
        return path.substr(path.find_last_of("/\\") +1);
    }else{
        return path.substr(path.find_last_of("/\\") +1, lastdot - lastdash - 1);
    }

}

std::string ExtractExtension( const std::string& path )
{
     if (path.find_last_of(".") == -1){
         return "";

     }else{
         return path.substr(path.find_last_of(".")) ;
     }

}


std::string ChangeExtension( const std::string& path, const std::string& ext )
{
    std::string filename = ExtractFilename( path );
    return ExtractDirectory( path ) +filename.substr( 0, filename.find_last_of( '.' )+1 ) +ext;
}

std::string fullFileName(FILE_PATH path_in){
    return (path_in.filepath + "/" + path_in.filename + "." + path_in.extension);
}

std::string FileNameExt(FILE_PATH path_in){
    return (path_in.filename + "." + path_in.extension);
}

void clean_file_path( FILE_PATH& path_in ){
    path_in.filename = "";
    path_in.filepath = "";
    path_in.extension = "";

}

unsigned int ifVaildPath(std::string file_name_in){
    struct stat info;
    if (stat (file_name_in.c_str(), &info) == 0){
        if (info.st_mode & S_IFDIR){
            return (ISFOLDER);
        }else if (info.st_mode & S_IFREG){
            return (ISFILE);
        }
    }else{
        std::cout<<"invalid_path"<<std::endl;
        return (INVALID_PATH);
    }

}

unsigned int ifVaildPath(std::string file_name_in, FILE_PATH& path_in){
    struct stat info;
    if (stat (file_name_in.c_str(), &info) == 0){
        if (info.st_mode & S_IFDIR){
            path_in.filepath = file_name_in;
            return (ISFOLDER);
        }else if (info.st_mode & S_IFREG){
            path_in.filepath = ExtractDirectory(file_name_in);
            path_in.filename = ExtractFilename(file_name_in);
            path_in.extension = ExtractExtension(file_name_in);
            return (ISFILE);
        }
    }else{
        std::cout<<"invalid_path"<<std::endl;
        return (INVALID_PATH);
    }

}

bool NewFolder(std::string folder_path){
    int lastdash = folder_path.find_last_of("/\\");
    std::string NewFolderName = folder_path.substr(lastdash+1);

    struct stat info;
    if (stat (folder_path.c_str(), &info) == 0){
        if (info.st_mode & S_IFDIR){
           std::cout<<"FolderName ["<< NewFolderName <<"] is exist!"<<std::endl;
           return (0);
         }
    }else{
        std::cout<< "NewFolderName = "<< NewFolderName<<"!"<<std::endl;
        //return mkdir(folder_path.c_str(), S_IWUSR | S_IWGRP | S_IWOTH);
        return mkdir(folder_path.c_str(), ACCESSPERMS);

    }

}

#endif
