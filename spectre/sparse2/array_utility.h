#pragma once

#include "range_zip.h"
#include <stdexcept>
#include <tuple>

constexpr unsigned normalize_axis(const int axis, const unsigned ndims) {
  if (axis < -ndims || axis >= ndims)
    throw std::invalid_argument{"Axis is out of bounds!"};
  return (axis < 0) ? (ndims + axis) : axis;
}

template <typename Axis, typename Array>
constexpr decltype(auto) shape(Array&& array) {
  std::get<Axis>(array.shape());
}

/******************************************************************************/

template <size_t... Indices, typename... Ts>
decltype(auto) extract(std::tuple<Ts...>& tuple) {
  return std::tuple{std::get<Indices>(tuple)...};
}

template <size_t... Indices, typename... Ts>
decltype(auto) extract(std::tuple<Ts...>&& tuple) {
  return std::tuple{std::get<Indices>(std::move(tuple))...};
}

template <size_t... Indices, typename... Ts>
decltype(auto) extract(const std::tuple<Ts...>& tuple) {
  return std::tuple{std::get<Indices>(tuple)...};
}

template <size_t... Indices, typename... Ts>
decltype(auto) extract(const std::tuple<Ts...>&& tuple) {
  return std::tuple{std::get<Indices>(std::move(tuple))...};
}

template <size_t... Indices, typename... Ts>
decltype(auto) extract(range::proxy_zip<Ts...>& proxy) {
  return range::proxy_zip{std::get<Indices>(proxy)...};
}

template <size_t... Indices, typename... Ts>
decltype(auto) extract(range::proxy_zip<Ts...>&& proxy) {
  return range::proxy_zip{std::get<Indices>(std::move(proxy))...};
}

template <size_t... Indices, typename... Ts>
decltype(auto) extract(const range::proxy_zip<Ts...>& proxy) {
  return range::proxy_zip{std::get<Indices>(proxy)...};
}

template <size_t... Indices, typename... Ts>
decltype(auto) extract(const range::proxy_zip<Ts...>&& proxy) {
  return range::proxy_zip{std::get<Indices>(std::move(proxy))...};
}

/******************************************************************************/

template <size_t Index, typename A, typename B>
decltype(auto) extract_except(std::tuple<A, B>& tuple) {
  return extract<1 - Index>(tuple);
}

template <size_t Index, typename A, typename B>
decltype(auto) extract_except(std::tuple<A, B>&& tuple) {
  return extract<1 - Index>(std::move(tuple));
}

template <size_t Index, typename A, typename B>
decltype(auto) extract_except(const std::tuple<A, B>& tuple) {
  return extract<1 - Index>(tuple);
}

template <size_t Index, typename A, typename B>
decltype(auto) extract_except(const std::tuple<A, B>&& tuple) {
  return extract<1 - Index>(std::move(tuple));
}

template <size_t Index, typename A, typename B>
decltype(auto) extract_except(range::proxy_zip<A, B>& proxy) {
  return extract<1 - Index>(proxy);
}

template <size_t Index, typename A, typename B>
decltype(auto) extract_except(range::proxy_zip<A, B>&& proxy) {
  return extract<1 - Index>(std::move(proxy));
}

template <size_t Index, typename A, typename B>
decltype(auto) extract_except(const range::proxy_zip<A, B>& proxy) {
  return extract<1 - Index>(proxy);
}

template <size_t Index, typename A, typename B>
decltype(auto) extract_except(const range::proxy_zip<A, B>&& proxy) {
  return extract<1 - Index>(std::move(proxy));
}

/******************************************************************************/