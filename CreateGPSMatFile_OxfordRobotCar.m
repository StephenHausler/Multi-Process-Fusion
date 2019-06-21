

clear variables
load('D:/Windows/oxford-data/2014-12-09-13-21-02/timestamps.mat');
load('D:/Windows/oxford-data/2014-12-09-13-21-02/RefGPSdata.mat');
load('D:/Windows/oxford-data/2014-12-10-18-10-50/timestamps.mat');  %query traverse images
load('D:/Windows/oxford-data/2014-12-10-18-10-50/QueryGPSdata.mat');
totalImagesR = 1866;
totalImagesQ = 2025;
imstart_R = 400;
imstart_Q = 1750;
GPSMatrix = zeros(totalImagesQ);

for ii = 1:totalImagesQ
    QueryTimeStamp = QueryTimeStampArray(((ii-1)*3)+imstart_Q);
    %use index to grab timestamp from GPS file
    %search gps.csv for nearest line/closest match for TimeStamp, then extract
    %latitude and longitude.
    %QueryGPSdata is a 'sorted list' so perform binary search
    L = 1; R = length(QueryGPSdata);
    while L <= R
        m = floor((L+R)/2);
        if QueryGPSdata(m,1) < QueryTimeStamp
            L = m+1;
        elseif QueryGPSdata(m,1) > QueryTimeStamp
            R = m-1;
        else
            QueryPos = m;
            break
        end
    end
    QueryPos = m;
    %then return corresponding GPS coordinate
    QueryLat(ii) = QueryGPSdata(QueryPos,2);
    QueryLong(ii) = QueryGPSdata(QueryPos,3);
end    
for j = 1:totalImagesR
    %grab index for matched image number
    %use index to grab timestamp
    ReferenceTimeStamp = ReferenceTimeStampArray(((j-1)*3)+imstart_R);
    %then search reference traverse GT file for identical/nearest timestamp
    L = 1; R = length(RefGPSdata);
    while L <= R
        m = floor((L+R)/2);
        if RefGPSdata(m,1) < ReferenceTimeStamp
            L = m+1;
        elseif RefGPSdata(m,1) > ReferenceTimeStamp
            R = m-1;
        else
            RefPos = m;
            break
        end
    end
    RefPos = m;
    %then return corresponding GPS coordinate
    RefLat(j) = RefGPSdata(RefPos,2); %col 2 is Lat
    RefLong(j) = RefGPSdata(RefPos,3); %col 3 is Long
end
for ii = 1:totalImagesQ
    for j = 1:totalImagesR
        %compare the two GPS coordinates with a GT tolerance of 40 meters
        d(j) = GPS2Meters(QueryLat(ii),QueryLong(ii),RefLat(j),RefLong(j));
        if d(j) <= 30
            GPSMatrix(j,ii) = 1;
        end
    end
    clear d
end

%construct Sparse GT matrix
GPSMatrix = sparse(GPSMatrix);
figure
imagesc(GPSMatrix)
save('OxfordRobotCar_GPSMatrix_30m','GPSMatrix')




%Function to calculate meters between two GPS coordinates.
function d = GPS2Meters(lat1,long1,lat2,long2)
    R = 6378.137;   %radius of earth in km
    dLat = lat2*(pi/180) - lat1*(pi/180);
    dLon = long2*(pi/180) - long1*(pi/180);
    a = sin(dLat/2)*sin(dLat/2) + cos(lat1*(pi/180))*cos(lat2*(pi/180))*sin(dLon/2)*sin(dLon/2);
    c = 2*atan2(sqrt(a),sqrt(1-a));
    d = R*c;
    d = d*1000;
end

