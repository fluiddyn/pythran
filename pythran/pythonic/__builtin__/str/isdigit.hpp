#ifndef PYTHONIC_BUILTIN_STR_ISDIGIT_HPP
#define PYTHONIC_BUILTIN_STR_ISDIGIT_HPP

#include "pythonic/include/__builtin__/str/isdigit.hpp"

#include "pythonic/types/str.hpp"
#include "pythonic/utils/functor.hpp"

PYTHONIC_NS_BEGIN

namespace __builtin__
{

  namespace str
  {

    bool isdigit(types::str const &s)
    {
      return !s.empty() && std::all_of(s.chars().begin(), s.chars().end(),
                                       (int (*)(int))std::isdigit);
    }
  }
}
PYTHONIC_NS_END
#endif
