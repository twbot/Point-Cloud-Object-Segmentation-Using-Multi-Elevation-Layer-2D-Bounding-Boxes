#include "../include/PlyReader.h"

using namespace PlyReader;

PlyProperty::PlyProperty(std::istream& is) : isList(false)
{
    parse_internal(is);
}

void PlyProperty::parse_internal(std::istream& is)
{
    std::string type;
    is >> type;
    if (type == "list")
    {
        std::string countType;
        is >> countType >> type;
        listType = property_type_from_string(countType);
        isList = true;
    }
    propertyType = property_type_from_string(type);
    is >> name;
}

PlyElement::PlyElement(std::istream& is)
{
    parse_internal(is);
}

void PlyElement::parse_internal(std::istream& is)
{
    is >> name >> size;
}

PlyFile::PlyFile(std::istream& is)
{
    if (!parse_header(is))
    {
        throw std::runtime_error("file is not ply or encounted junk in header");
    }

}

bool PlyFile::parse_header(std::istream& is)
{
    std::string line;
    while (std::getline(is, line))
    {
        std::istringstream ls(line);
        std::string token;
        ls >> token;
        if (token == "ply" || token == "PLY" || token == "")
        {
            continue;
        }
        else if (token == "comment")    read_header_text(line, ls, comments, 8);
        else if (token == "format")     read_header_format(ls);
        else if (token == "element")    read_header_element(ls);
        else if (token == "property")   read_header_property(ls);
        else if (token == "obj_info")   read_header_text(line, ls, objInfo, 9);
        else if (token == "end_header") break;
        else return false;
    }
    return true;
}

void PlyFile::read_header_text(std::string line, std::istream& is, std::vector<std::string>& place, int erase)
{
    place.push_back((erase > 0) ? line.erase(0, erase) : line);
}

void PlyFile::read_header_format(std::istream& is)
{
    std::string s;
    (is >> s);
    if (s == "binary_little_endian") isBinary = true;
    else if (s == "binary_big_endian") isBinary = isBigEndian = true;
}

void PlyFile::read_header_element(std::istream& is)
{
    get_elements().emplace_back(is);
}

void PlyFile::read_header_property(std::istream& is)
{
    get_elements().back().properties.emplace_back(is);
}

size_t PlyFile::skip_property_binary(const PlyProperty& property, std::istream& is)
{
    static std::vector<char> skip(PropertyTable[property.propertyType].stride);
    if (property.isList)
    {
        size_t listSize = 0;
        size_t dummyCount = 0;
        read_property_binary(property.listType, &listSize, dummyCount, is);
        for (size_t i = 0; i < listSize; ++i) is.read(skip.data(), PropertyTable[property.propertyType].stride);
        return listSize;
    }
    else
    {
        is.read(skip.data(), PropertyTable[property.propertyType].stride);
        return 0;
    }
}

void PlyFile::skip_property_ascii(const PlyProperty& property, std::istream& is)
{
    std::string skip; 
    if (property.isList)
    {
        int listSize;
        is >> listSize;
        for (int i = 0; i < listSize; ++i) is >> skip;
    }
    else is >> skip;
}

void PlyFile::read_property_binary(PlyProperty::Type t, void* dest, size_t& destOffset, std::istream& is)
{
    static std::vector<char> src(PropertyTable[t].stride);
    is.read(src.data(), PropertyTable[t].stride);

    switch (t)
    {
    case PlyProperty::Type::INT8:       ply_cast<int8_t>(dest, src.data(), isBigEndian);        break;
    case PlyProperty::Type::UINT8:      ply_cast<uint8_t>(dest, src.data(), isBigEndian);       break;
    case PlyProperty::Type::INT16:      ply_cast<int16_t>(dest, src.data(), isBigEndian);       break;
    case PlyProperty::Type::UINT16:     ply_cast<uint16_t>(dest, src.data(), isBigEndian);      break;
    case PlyProperty::Type::INT32:      ply_cast<int32_t>(dest, src.data(), isBigEndian);       break;
    case PlyProperty::Type::UINT32:     ply_cast<uint32_t>(dest, src.data(), isBigEndian);      break;
    case PlyProperty::Type::FLOAT32:    ply_cast_float<float>(dest, src.data(), isBigEndian);   break;
    case PlyProperty::Type::FLOAT64:    ply_cast_double<double>(dest, src.data(), isBigEndian); break;
    case PlyProperty::Type::INVALID:    throw std::invalid_argument("invalid ply property");
    }
    destOffset += PropertyTable[t].stride;
}

void PlyFile::read_property_ascii(PlyProperty::Type t, void* dest, size_t& destOffset, std::istream& is)
{
    switch (t)
    {
    case PlyProperty::Type::INT8:       *((int8_t*)dest) = ply_read_ascii<int32_t>(is);        break;
    case PlyProperty::Type::UINT8:      *((uint8_t*)dest) = ply_read_ascii<uint32_t>(is);      break;
    case PlyProperty::Type::INT16:      ply_cast_ascii<int16_t>(dest, is);                      break;
    case PlyProperty::Type::UINT16:     ply_cast_ascii<uint16_t>(dest, is);                     break;
    case PlyProperty::Type::INT32:      ply_cast_ascii<int32_t>(dest, is);                      break;
    case PlyProperty::Type::UINT32:     ply_cast_ascii<uint32_t>(dest, is);                     break;
    case PlyProperty::Type::FLOAT32:    ply_cast_ascii<float>(dest, is);                        break;
    case PlyProperty::Type::FLOAT64:    ply_cast_ascii<double>(dest, is);                       break;
    case PlyProperty::Type::INVALID:    throw std::invalid_argument("invalid ply property");
    }
    destOffset += PropertyTable[t].stride;
}

void PlyFile::read(std::istream& is)
{
    read_internal(is);
}

void PlyFile::read_internal(std::istream& is)
{
    std::function<void(PlyProperty::Type t, void* dest, size_t& destOffset, std::istream& is)> read;
    std::function<void(const PlyProperty& property, std::istream& is)> skip;
    if (isBinary)
    {
        read = [&](PlyProperty::Type t, void* dest, size_t& destOffset, std::istream& _is) { read_property_binary(t, dest, destOffset, _is); };
        skip = [&](const PlyProperty& property, std::istream& _is) { skip_property_binary(property, _is); };
    }
    else
    {
        read = [&](PlyProperty::Type t, void* dest, size_t& destOffset, std::istream& _is) { read_property_ascii(t, dest, destOffset, _is); };
        skip = [&](const PlyProperty& property, std::istream& _is) { skip_property_ascii(property, _is); };
    }

    for (auto& element : get_elements())
    {
        if (std::find(requestedElements.begin(), requestedElements.end(), element.name) != requestedElements.end())
        {
            for (size_t count = 0; count < element.size; ++count)
            {
                for (auto& property : element.properties)
                {
                    if (auto& cursor = userDataTable[make_key(element.name, property.name)])
                    {
                        if (property.isList)
                        {
                            size_t listSize = 0;
                            size_t dummyCount = 0;
                            read(property.listType, &listSize, dummyCount, is);
                            if (cursor->realloc == false)
                            {
                                cursor->realloc = true;
                                resize_vector(property.propertyType, cursor->vector, listSize * element.size, cursor->data);
                            }
                            for (size_t i = 0; i < listSize; ++i)
                            {
                                read(property.propertyType, (cursor->data + cursor->offset), cursor->offset, is);
                            }
                        }
                        else
                        {
                            read(property.propertyType, (cursor->data + cursor->offset), cursor->offset, is);
                        }
                    }
                    else
                    {
                        skip(property, is);
                    }
                }
            }
        }
        else continue;
    }
}