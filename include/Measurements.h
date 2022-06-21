#pragma once

namespace Measurements
{	
	constexpr double Pi = 3.1415926535897931;
	constexpr double TwoPi = 2.0 * Pi;
	constexpr double FullCircle = 360.0;
	constexpr double DegToRad = Pi / 180.0;
	constexpr double RadToDeg = 180.0 / Pi;

	constexpr double FeetToMeters = 0.3048;
	constexpr double MetersToFeet = 3.28084;

	template<typename T, typename ValueType>
	class BaseScalarMeasurement
	{
	protected:
		ValueType _value = { 0 };

	public:
		T operator - () const { return T(-_value); }

		T operator + (const T & other) const { return T(_value + other._value); }
		T operator - (const T & other) const { return T(_value - other._value); }

		T & operator += (const T & other) { _value += other._value; return static_cast<T &>(*this); }
		T & operator -= (const T & other) { _value -= other._value; return static_cast<T &>(*this); }

		//
		// Multiply and divide _value by a scalar value:
		//

		T operator * (const ValueType scalar) const { return T(_value * scalar); }
		T operator / (const ValueType scalar) const { return T(_value / scalar); }

		T & operator *= (const ValueType scalar) { _value *= scalar; return static_cast<T &>(*this); }
		T & operator /= (const ValueType scalar) { _value /= scalar; return static_cast<T &>(*this); }


		//
		// Expose the built-in comparison operators:
		//

		bool operator == (const T & other) const { return _value == other._value; }
		bool operator != (const T & other) const { return _value != other._value; }
		bool operator <= (const T & other) const { return _value <= other._value; }
		bool operator >= (const T & other) const { return _value >= other._value; }
		bool operator  < (const T & other) const { return _value < other._value; }
		bool operator  > (const T & other) const { return _value > other._value; }

		// Exposing stream insertion 
		friend std::ostream & operator << (std::ostream &o, BaseScalarMeasurement &d) { o << d._value; return o; }


		const ValueType &Get() const
		{
			return _value;
		}
	};

	/*
	*
	* Degrees and Radians as measurements and type safety
	*
	*/
	class Radians;
	class Degrees;
	class Degrees_f;

	template<typename T, typename ValueType>
	class CircularConstrained : public BaseScalarMeasurement<T, ValueType>
	{
	public:
		//0 - FullCircle amount EX: 0 - 360, 0 - 2*PI
		T Constrained() const
		{
			ValueType newValue = (ValueType) std::fmod(this->_value, T::ModuloValue);
			if (newValue < T::ModuloValue) newValue += T::ModuloValue;
			return T(newValue);
		}

		T Difference(const T &InValue)
		{
			auto halfMod = (ValueType)(T::ModuloValue / 2.0);
			return T(halfMod - std::fabs(std::fmod(std::fabs(T(InValue - *(T*)this).Get()), T::ModuloValue) - halfMod));
		}
		
		bool IsEqual(const T &InValue, double epsilon = 0.001)
		{
			return std::fabs((Constrained() - InValue.Constrained()).Get()) < epsilon;
		}
	};

	class Radians : public CircularConstrained<Radians, double>
	{
	public:
		static constexpr double const ModuloValue = TwoPi;
		
		Radians() {}
		explicit Radians(double inValue)
		{
			_value = inValue;
		}

		Radians(const Degrees & degrees);
	};

	class Degrees : public CircularConstrained<Degrees, double>
	{
	public:		
		static constexpr double const ModuloValue = FullCircle;
		
		Degrees() {}
		explicit Degrees(double inValue)
		{
			_value = inValue;
		}
		Degrees(const Radians & radians);	
		Degrees(const Degrees_f & degrees);
	};

	template<>
	inline StorageInterface& operator<< <Degrees>(StorageInterface &StorageInterface, const Degrees& m) {
		StorageInterface << m.Get();
		return StorageInterface;
	}
	template<>
	inline StorageInterface& operator>> (StorageInterface &StorageInterface,  Degrees& m) {
		double temp;
		StorageInterface >> temp;
		m = Degrees(temp);
		return StorageInterface;
	}

	class Degrees_f : public CircularConstrained<Degrees_f, float>
	{		
	public:
		static constexpr float const ModuloValue = FullCircle;
		
		Degrees_f() {}
		explicit Degrees_f(float inValue)
		{
			_value = inValue;
		}
		Degrees_f(const Radians & radians);
	};
	template<>
	inline StorageInterface& operator<< <Degrees_f>(StorageInterface &StorageInterface, const Degrees_f& m) {
		StorageInterface << m.Get();
		return StorageInterface;
	}
	template<>
	inline StorageInterface& operator>> (StorageInterface &StorageInterface, Degrees_f& m) {
		float temp;
		StorageInterface >> temp;
		m = Degrees_f(temp);
		return StorageInterface;
	}
	
	inline Radians::Radians(const Degrees & degrees)
	{
		_value = degrees.Get() * DegToRad;
	}

	inline Degrees::Degrees(const Radians & radians)
	{
		_value = radians.Get() * RadToDeg;
	}

	inline Degrees::Degrees(const Degrees_f &degrees)
	{
		_value = degrees.Get();
	}

	inline Degrees_f::Degrees_f(const Radians & radians) {
		_value = (float)(radians.Get() * RadToDeg);
	}


	// User defined suffix `_rad` for literals with type `Radians` (C++11).
	inline Radians operator "" _rad(long double radians)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Radians(static_cast<double>(radians));
	}
	inline Radians operator "" _rad(unsigned long long int radians)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Radians(static_cast<double>(radians));
	}

	inline Degrees operator "" _deg(long double degrees)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Degrees(static_cast<double>(degrees));
	}
	inline Degrees operator "" _deg(unsigned long long int degrees)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Degrees(static_cast<double>(degrees));
	}

	inline Degrees_f operator "" _degf(unsigned long long int degrees)
	{
		return Degrees_f(static_cast<float>(degrees));
	}

	inline Degrees_f operator "" _degf(long double degrees){
		// Note: The standard requires the input parameter to be `long double`!
		return Degrees_f(static_cast<float>(degrees));
	}

	/*
	*
	* Feet and Meters as measurements and type safety
	*
	*/
	class Feet;
	class Meters;

	class Feet : public BaseScalarMeasurement<Feet, double>
	{
	public:
		Feet() { }
		explicit Feet(double inValue)
		{
			_value = inValue;
		}
		Feet(const Meters & degrees);
	};

	class Meters : public BaseScalarMeasurement<Meters, double>
	{
	public:
		Meters() { }
		Meters(double inValue)
		{
			_value = inValue;
		}
		Meters(const Feet & radians);
	};
	template<>
	inline StorageInterface& operator<< <Meters>(StorageInterface &StorageInterface, const Meters& m) {
		StorageInterface << m.Get();
		return StorageInterface;
	}
	template<>
	inline StorageInterface& operator>> (StorageInterface &StorageInterface, Meters& m) {
		double temp;
		StorageInterface >> temp;
		m = Meters(temp);
		return StorageInterface;
	}

	inline Feet::Feet(const Meters & InValue)
	{
		_value = InValue.Get() * MetersToFeet;
	}
	inline Meters::Meters(const Feet &InValue)
	{
		_value = InValue.Get() * FeetToMeters;		
	}


	// User defined suffix `_rad` for literals with type `Radians` (C++11).
	inline Feet operator "" _ft(long double radians)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Feet(static_cast<double>(radians));
	}
	inline Feet operator "" _ft(unsigned long long int radians)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Feet(static_cast<double>(radians));
	}

	inline Meters operator "" _m(long double degrees)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Meters(static_cast<double>(degrees));
	}
	inline Meters operator "" _m(unsigned long long int degrees)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Meters(static_cast<double>(degrees));
	}
	inline Meters operator "" _km(long double degrees)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Meters(static_cast<double>(degrees) * 1000.0);
	}
	inline Meters operator "" _km(unsigned long long int degrees)
	{
		// Note: The standard requires the input parameter to be `long double`!
		return Meters(static_cast<double>(degrees) * 1000.0);
	}

}