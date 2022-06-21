#pragma once

#include "Measurements.h"
#include <Eigen/Core>
#include <algorithm>

using namespace Measurements;

namespace Coordinates
{
	using Matrix2x2 = Eigen::Matrix< double, 2, 2, Eigen::RowMajor >;
	using Matrix3x3 = Eigen::Matrix< double, 3, 3, Eigen::RowMajor >;
	using Matrix4x4 = Eigen::Matrix< double, 4, 4, Eigen::RowMajor >;
	
	using Vector2 = Eigen::Matrix< double, 1, 2, Eigen::RowMajor >;
	using Vector3 = Eigen::Matrix< double, 1, 3, Eigen::RowMajor >;
	using Vector4 = Eigen::Matrix< double, 1, 4, Eigen::RowMajor >;

	using Vector2i = Eigen::Matrix< int32_t, 1, 2, Eigen::RowMajor >;
	using Vector3i = Eigen::Matrix< int32_t, 1, 3, Eigen::RowMajor >;
	using Vector4i = Eigen::Matrix< int32_t, 1, 4, Eigen::RowMajor >;

	using MeterVec2 = Eigen::Matrix< Meters, 1, 2, Eigen::RowMajor>;
	using MeterVec3 = Eigen::Matrix< Meters, 1, 3, Eigen::RowMajor >;
	using MeterVec4 = Eigen::Matrix< Meters, 1, 4, Eigen::RowMajor >;
	
	using FeetVec2 = Eigen::Matrix< Feet, 1, 2, Eigen::RowMajor >;
	using FeetVec3 = Eigen::Matrix< Feet, 1, 3, Eigen::RowMajor >;
	using FeetVec4 = Eigen::Matrix< Feet, 1, 4, Eigen::RowMajor >;
		

	inline const double RR_PI = 3.14159265359;
	inline const double RR_feetToMeters = 0.3048;
	inline const double RR_metersToFeet = (1.0 / RR_feetToMeters);
	inline const double RR_milesToFeet = 5280.0;
	inline const double RR_feetToMiles = (1.0 / RR_milesToFeet);
	inline const double RR_degToRad = (RR_PI / 180.0);
	inline const double RR_radToDeg = (1.0 / RR_degToRad);
	inline const double RR_dtf = 365565.0; // 364320.0;
	inline const double RR_ftd = (1.0 / RR_dtf);
	inline const double RR_ReFeet = 20902231.0;	//radius of the earth in feet, assuming a spherical model


	struct EigenHelpers
	{
	public:

		/**
		 * Creates a rotation matrix about an axis a yaw (heading) attitude value
		 * @param angleInDegrees Yaw angle in degress
		 * @param axisOfRotation axis about which the rotation will occur
		 * @return the resulting Matrix3x3
		 */
		static Matrix3x3 createRotationMatrix(Degrees angleInDegrees, int axisOfRotation);

		/**
		 * Creates a rotation matrix about an axis a yaw (heading) attitude value
		 * @param angleInDegrees Yaw angle in degress
		 * @param axisOfRotation axis about which the rotation will occur
		 * @return the resulting Matrix3x3
		 */
		template <typename T>
		static Eigen::Matrix<T, 3, 3> createRotationMatrix_t(T angleInRadians, int axisOfRotation)
		{
			Eigen::Matrix<T, 3, 3> mat = Eigen::Matrix<T, 3, 3>::Identity();
			T c = cos(angleInRadians);
			T s = sin(angleInRadians);
			std::vector<int> axes = { 0,1,2 };
			axes.erase(axes.begin() + axisOfRotation);
			mat(axes[0], axes[0]) = c;  mat(axes[0], axes[1]) = -s;
			mat(axes[1], axes[0]) = s;  mat(axes[1], axes[1]) = c;
			return mat;
		};

		/**
		 * Creates ENA-directed Body-to-Earth Direction Cosine matrix based upon a
		 * transformation of
		 * Roll, Pitch, and Yaw (in that order)
		 * @param roll Angle in degrees
		 * @param pitch Angle in degrees
		 * @param yaw Angle in degrees
		 * @return Resulting ENA-directed Body-to-Earth Direction Cosine matrix
		 */
		static Matrix3x3 buildDirCosV3(Degrees roll, Degrees pitch, Degrees yaw);
		
		// the inverse of buildDirCosV3. Returns ypr
		static void getYPRV3(double *ypr, Matrix3x3 &m);

		static Matrix3x3 toV4(const Matrix3x3 &m);
		static Matrix3x3 toV3(const Matrix3x3 &m);

		/**
		 * Creates ENA-directed Body-to-Earth Direction Cosine matrix based upon a
		 * transformation of
		 * Roll, Pitch, and Yaw (in that order)
		 * @param roll Angle in degrees
		 * @param pitch Angle in degrees
		 * @param yaw Angle in degrees
		 * @return Resulting ENA-directed Body-to-Earth Direction Cosine matrix
		 */
		static Matrix3x3 buildDirCosV4(Degrees roll, Degrees pitch, Degrees yaw);


		/**
		 * Corrects from the mount to the camera with Yaw, Pitch, and Roll (in that order)
		 * @param yaw Angle in degrees
		 * @param roll Angle in degrees
		 * @param pitch Angle in degrees
		 * @return Resulting camera correction matrix
		 */
		static Matrix3x3 buildMountMatrixYPRDegreesV3(double *ypr) {
			double cy = cos(ypr[0] * DegToRad), sy = sin(ypr[0] * DegToRad);
			double cp = cos(ypr[2] * DegToRad), sp = sin(ypr[2] * DegToRad);
			double cr = cos(ypr[1] * DegToRad), sr = sin(ypr[1] * DegToRad);
			static Matrix3x3 m_off;
			m_off <<	 cr*cy,          -cr*sy,          sr,
						 cp*sy-cy*sp*sr,  cp*cy+sp*sr*sy, cr*sp,
						-sp*sy-cp*cy*sr,  cp*sr*sy-cy*sp, cp*cr;
			return m_off;
		}

		static bool useLegacyXform() { return false; }

	};

	template<typename T, int32_t ArraySize>
	class ArrayVariableArgs {
	protected:
		std::array<T, ArraySize> coordinate = { 0 };

	public:
		ArrayVariableArgs& operator = (const std::array<T, ArraySize>& new_coordinate) {
			coordinate = new_coordinate;
			return *this;
		}

		double operator [](int i) const {
			return coordinate[i];
		}

		double &operator [](int i) {
			return coordinate[i];
		}

		std::array<T, ArraySize> coordinates() {
			return coordinate;
		}

		std::string print()const {
			std::stringstream ss;
			for (int i = 0; i < ArraySize; i++) {
				ss << coordinate[i] << " ";
			}
			return ss.str();
		}		
	};

	template <typename T, int32_t ArraySize>
	inline StorageInterface& operator<< (StorageInterface &StorageInterface, const std::array<T, ArraySize>& InData)
	{
		for (int32_t Iter = 0; Iter < ArraySize; ++Iter)
		{
			StorageInterface << InData[Iter];
		}
		return StorageInterface;
	}

	template <typename T, int32_t ArraySize>
	inline StorageInterface& operator>> (StorageInterface &StorageInterface, std::array<T, ArraySize>& OutData)
	{
		for (int32_t Iter = 0; Iter < ArraySize; ++Iter)
		{
			StorageInterface >> OutData[Iter];
		}
		return StorageInterface;
	}

	//==============================================================================

	struct ECEF;
	struct LLA;
	struct UTM;
	struct ENA;
	struct LatLong;

	enum ECardinalDirections {
		SOUTH = 0,
		WEST = 1,
		NORTH = 2,
		EAST = 3,
	};

	class A_SWNE : public ArrayVariableArgs<double, 4> {
		using ParentArray = ArrayVariableArgs<double, 4>;
		
	public:
		double degrees_EW();
		double degrees_NS();
		double average_latitude();
		double average_longitude();

		A_SWNE() = default;
		A_SWNE GetSWNE();
		/** returns a vector of LLA (NW, SW, SE, NE)
		*/
		std::vector<LLA> GetPoints();
		A_SWNE(double south, double west, double north, double east);

		A_SWNE& operator = (const std::array<double, 4>& new_coordinate);
		double return_area_in_solid_degrees();
		double return_intersection_fraction(A_SWNE& coordinate_2);
		std::array<double, 2> return_center();
		bool equates(const A_SWNE& coordinate_2) const;
		bool contains(const A_SWNE& coordinate_2) const;
		bool contains(LLA& lla);
		bool isEmpty() const;
		void AssimilatePoint(LLA point);
		void Assimilate_SWNE(A_SWNE& coordiante_2);

		/**valid means that our box contains the other one
		 * and that the centers are no more than 10% of the total size away. */
		bool isValidBound(A_SWNE boundingBox);
		std::array<double, 4> return_intersection(A_SWNE& coordinate_2);
		std::array<double, 4> return_normal_inner(A_SWNE& coordinate_2);

 
		static A_SWNE fromCenterAndSize(LLA &centerLLA, Meters size[2]);
		static A_SWNE fromCenterAndSize(LLA &centerLLA, Meters sizeEW, Meters sizeNS);

		std::string ToString() const; //fmm: possible wrong naming convention: methods that don't return void
									  //should begin with a lowercase letter.

		friend inline StorageInterface& operator<< (StorageInterface &StorageInterface, const A_SWNE& InData)
		{
			StorageInterface << InData.coordinate;
			return StorageInterface;
		}

		friend inline StorageInterface& operator>> (StorageInterface &StorageInterface, A_SWNE& OutData)
		{
			StorageInterface >> OutData.coordinate;
			return StorageInterface;
		}
	};

	

	//==============================================================================

	struct ECEF
	{
		MeterVec3 _coord;

		ECEF() = default;
		ECEF(Meters x, Meters y, Meters z);
		ECEF(double x, double y, double z);
		ECEF(const MeterVec3 & coordinate);
		ECEF(const ECEF & otherECEF);
		ECEF(const LLA &fromGeoDesic);
		ECEF(const UTM & fromUTM);

		void GetPosition(Meters &oX, Meters &oY, Meters &oZ) const;
		void SetPosition(const Meters &iX, const Meters &iY, const Meters &iZ);
		void GetMetersAsVector3(Vector3 &oVector) const;

		LLA convertLLA()const;							
		UTM convertUTM()const;							
		ENA convertENA(const LLA & refLLA)const;		//not tested
		//Meters geoidHeight()const;					//not tested //needs to be implemented

		std::string ToString() const;
	};
		
	struct LLA
	{
		Degrees _latitude;
		Degrees _longitude;
		Meters _altitude;

		LLA() = default; 
		LLA(Degrees InLatitude, Degrees InLongtitude, Meters InAltitude);
		LLA(double inLatitude, double inLongitude, double inAltitude);
		LLA(double *inLLADoubles);
		LLA(const ECEF &fromGeoCentric);
		LLA(const LLA &otherLLA);
		LLA(const UTM & fromUTM);

		LLA operator+(const ENA &ena);
		LLA operator-(const ENA &ena);
		bool operator==(const LLA& InCompare) const;

		void GetPosition(Degrees &oLat, Degrees &oLong, Meters &oA) const;
		void GetRectangle()const;												//needs to be implemented
		void Get(double *lla);

		Meters geoidHeight()const;								//not tested
		ECEF convertECEF()const;								//not tested
		ENA convertENA(const LLA & refLLA)const;
		UTM convertUTM()const;									//not tested
		LatLong convertLL()const;		
		
		std::string ToString() const { return std::string_format("%f(lat) %f(long) %fm(altitude)", _latitude.Get(), _longitude.Get(), _altitude.Get()); }

		Matrix4x4 GetECEFTransformation() const;
	};

	struct UTM
	{	
		Meters _easting;
		Meters _northing;
		int32_t _zone;
		Meters _altitude;

		UTM() = default;
		UTM(Meters InEasting, Meters InNorthing, Meters altitude, int32_t InZone);
		UTM(double InEasting, double InNorthing, double altitude, int32_t InZone);
		UTM(const ECEF &fromGeoCentric);
		UTM(const LLA & fromLLA);
		UTM(const UTM & otherUTM);

		LLA convertLLA()const;							//not tested
		ECEF convertECEF()const;						//not tested
		ENA convertENA(const LLA & reflLLA)const;		//not tested

		std::string ToString() const { return std::string_format("%fm(easting) %fm(northing) %f(zone)", _easting.Get(), _northing.Get(), _zone); }
	};

	struct ENA {
		MeterVec3 _coordinate;

		ENA();
		ENA(Meters e, Meters n, Meters a);
		ENA(double e, double n, double a);
		ENA(MeterVec3 coordinate);
		ENA(const ENA &otherENA);

		LLA convertLLA(const LLA & initialLLA)const;
		void convertLLA(double *lla, double initialLLA[3])const;
		void convertLLA(double *lla, LLA & initialLLA)const;
		Meters distance();
	};

	struct LatLong
	{
		Degrees latitude;
		Degrees longitude;

		LatLong() = default;
		LatLong(const Degrees &InLat, const Degrees &InLong) : latitude(InLat), longitude(InLong) {}
		LatLong(double InLat, double InLong);

		LatLong operator - () const;
		LatLong operator / (double InValue) const
		{
			//this is kinda dumb
			return LatLong(latitude / InValue, longitude / InValue);
		}
		LatLong operator + (const LatLong & other) const;
		LatLong operator - (const LatLong & other) const;

		LLA convertLLA()const;

		std::string ToString();
	};
	//================================================================


	inline LatLong maxmulticomp(const LatLong &InA, const LatLong &InB)
	{
		return LatLong(std::max(InA.latitude, InB.latitude), std::max(InA.longitude, InB.longitude));
	}

	inline Vector2 maxmulticomp(const Vector2 &InA, const Vector2 &InB)
	{
		return Vector2(std::max(InA[0], InB[0]),std::max(InA[1], InB[1]));
	}

	inline Vector3 maxmulticomp(const Vector3 &InA, const Vector3 &InB)
	{
		return Vector3(std::max(InA[0], InB[0]), std::max(InA[1], InB[1]), std::max(InA[2], InB[2]));
	}

	inline Vector4 maxmulticomp(const Vector4 &InA, const Vector4 &InB)
	{
		return Vector4(std::max(InA[0], InB[0]), std::max(InA[1], InB[1]), std::max(InA[2], InB[2]), std::max(InA[3], InB[3]));
	}
	
	inline LatLong minmulticomp(const LatLong &InA, const LatLong &InB)
	{
		return LatLong(std::min(InA.latitude, InB.latitude), std::min(InA.longitude, InB.longitude));
	}

	inline Vector2 minmulticomp(const Vector2 &InA, const Vector2 &InB)
	{
		return Vector2(std::min(InA[0], InB[0]), std::min(InA[1], InB[1]));
	}

	inline Vector3 minmulticomp(const Vector3 &InA, const Vector3 &InB)
	{
		return Vector3(std::min(InA[0], InB[0]), std::min(InA[1], InB[1]), std::min(InA[2], InB[2]));
	}

	inline Vector4 minmulticomp(const Vector4 &InA, const Vector4 &InB)
	{
		return Vector4(std::min(InA[0], InB[0]), std::min(InA[1], InB[1]), std::min(InA[2], InB[2]), std::min(InA[3], InB[3]));
	}

	template<typename T>
	class TBoundingBox
	{
	protected:
		bool bValid = false;
		T _min;
		T _max;

	public:
		TBoundingBox() {}
		TBoundingBox(const T &InMin, const T &InMax) : _min(InMin), _max(InMax)
		{
			bValid = true;
		}

		T GetMin() const
		{
			return _min;
		}
		T GetMax() const
		{
			return _max;
		}

		bool IsValid() const
		{
			return bValid;
		}

		T GetCenter() const
		{
			return (_max + _min) / 2.0;
		}

		T GetExtents() const
		{
			return (_max - GetCenter());
		}

		void operator+=(const T &InPosition)
		{
			if (bValid == false)
			{
				_max = InPosition;
				_min = InPosition;
				bValid = true;
				return;
			}

			_max = maxmulticomp(_max, InPosition);
			_min = minmulticomp(_min, InPosition);
		}
	};
		
	using BoundingBoxVec2 = TBoundingBox< Vector2 >;
	using BoundingBoxVec3 = TBoundingBox< Vector3 >;
	using BoundingBoxVec4 = TBoundingBox< Vector4 >;

	using BoundingBox = BoundingBoxVec3;

	class LatLongBoundingBox : public TBoundingBox< LatLong >
	{
	public:
		LatLongBoundingBox();
		LatLongBoundingBox(const LatLong &InMin, const LatLong &InMax);
		LatLongBoundingBox(const class A_SWNE &InSWNE);

		LatLong GetNorthWest() const;
		LatLong GetSouthEast() const;
		A_SWNE GetSWNE() const;

		LatLongBoundingBox Shift(LatLong InShift) const;

		//
		BoundingBoxVec2 GetMetersBounds(const Degrees &InLatitude) const;

		std::string ToString();
	};

	//fmm: old manual serialization for the LLA. Move to reflection.
	template<>
	inline StorageInterface& operator<< <LLA>(StorageInterface &StorageInterface, const LLA& m) {
		StorageInterface << m._latitude;
		StorageInterface << m._longitude;
		StorageInterface << m._altitude;
		return StorageInterface;
	}
	template<>
	inline StorageInterface& operator>> (StorageInterface &StorageInterface, LLA& m) {
		StorageInterface >> m._latitude;
		StorageInterface >> m._longitude;
		StorageInterface >> m._altitude;
		return StorageInterface;
	}	

	Degrees return_degrees_latitude_from_distance(Feet distance);						//checked

	Degrees return_degrees_latitute_from_distance(Meters distance);						//checked

	Degrees return_degrees_longitude_from_distance(Feet distance, Degrees latitude);	//checked

	Feet return_distance_from_degrees_latitude(Degrees distance);						//checked 

	Feet return_distance_from_degrees_longitude(Degrees distance, Degrees latitude);	//checked
	
	class RRCoordinates {
	public:
		static int test();
	};


	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	inline StorageInterface& operator<< (StorageInterface &StorageInterface, const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >& data)
	{
		for (uint32_t col = 0; col < data.cols(); ++col)
		{
			for (uint32_t row = 0; row < data.rows(); ++row)
			{
				StorageInterface << data(row, col);
			}
		}

		return StorageInterface;
	}
	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	inline StorageInterface& operator>> (StorageInterface &StorageInterface, Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols >& data)
	{
		for (uint32_t col = 0; col < data.cols(); ++col)
		{
			for (uint32_t row = 0; row < data.rows(); ++row)
			{
				StorageInterface >> data(row, col);
			}
		}

		return StorageInterface;
	}
}