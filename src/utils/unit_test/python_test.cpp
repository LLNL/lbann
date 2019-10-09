// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/python.hpp>

#ifdef LBANN_HAS_PYTHON
TEST_CASE ("Testing the embedded Python session", "[python][utilities]") {

  /// @todo Test finalization and reinitialization
  SECTION("Before initialization") {
    // Check twice to make sure interacting doesn't initialize Python
    CHECK_FALSE(lbann::python::is_active());
    CHECK_FALSE(lbann::python::is_active());
  }

  SECTION("Initialization") {
    REQUIRE_NOTHROW(lbann::python::initialize());
    REQUIRE(lbann::python::is_active());
  }

  SECTION("Global interpreter lock") {
    SECTION("Acquire GIL once") {
      std::unique_ptr<lbann::python::global_interpreter_lock> gil;
      REQUIRE_NOTHROW(gil.reset(new lbann::python::global_interpreter_lock()));
      REQUIRE_NOTHROW(gil.reset());
    }
    SECTION("Acquire GIL recursively") {
      std::unique_ptr<lbann::python::global_interpreter_lock> gil1, gil2, gil3;
      REQUIRE_NOTHROW(gil1.reset(new lbann::python::global_interpreter_lock()));
      REQUIRE_NOTHROW(gil2.reset(new lbann::python::global_interpreter_lock()));
      REQUIRE_NOTHROW(gil3.reset(new lbann::python::global_interpreter_lock()));
      REQUIRE_NOTHROW(gil3.reset());
      REQUIRE_NOTHROW(gil2.reset());
      REQUIRE_NOTHROW(gil1.reset());
    }
  }

  SECTION("Python error checking") {
    lbann::python::global_interpreter_lock gil;
    REQUIRE_NOTHROW(lbann::python::check_error());
    REQUIRE_THROWS(lbann::python::check_error(true));

    // Syntax error
    /// @todo Handle exception gracefully
    // auto main = PyImport_ImportModule("__main__");
    // auto x = PyObject_GetAttrString(main, "_bogus_variable_name({)}))");
    // REQUIRE(x == nullptr);
    //REQUIRE_THROWS(lbann::python::check_error());
    // Py_DECREF(main);
    REQUIRE_NOTHROW(lbann::python::check_error());

  }

  SECTION("Python object") {
    lbann::python::global_interpreter_lock gil;

    SECTION("Default constructor") {
      std::unique_ptr<lbann::python::object> obj;
      REQUIRE_NOTHROW(obj.reset(new lbann::python::object()));
      REQUIRE(*obj == nullptr);
      REQUIRE_NOTHROW(obj.reset());
    }

    SECTION("PyObject* constructor") {
      PyObject* ptr = Py_BuildValue("(i,d,s)", 987, 6.54, "321");
      REQUIRE(ptr != nullptr);
      Py_INCREF(ptr);
      REQUIRE(Py_REFCNT(ptr) == 2);
      std::unique_ptr<lbann::python::object> obj;
      REQUIRE_NOTHROW(obj.reset(new lbann::python::object(ptr)));
      REQUIRE(Py_REFCNT(ptr) == 2);
      REQUIRE_NOTHROW(obj.reset());
      REQUIRE(Py_REFCNT(ptr) == 1);
      Py_DECREF(ptr);
      REQUIRE_NOTHROW(lbann::python::check_error());
    }

    SECTION("PyObject* access functions") {
      PyObject* ptr = Py_BuildValue("(i,d,s)", 12, 3.4, "56");
      REQUIRE(ptr != nullptr);
      Py_INCREF(ptr);
      std::unique_ptr<lbann::python::object> obj;
      REQUIRE_NOTHROW(obj.reset(new lbann::python::object(ptr)));
      REQUIRE(Py_REFCNT(ptr) == 2);
      REQUIRE(obj->get() == ptr);
      REQUIRE(Py_REFCNT(ptr) == 2);
      REQUIRE(const_cast<const lbann::python::object&>(*obj).get() == ptr);
      REQUIRE(Py_REFCNT(ptr) == 2);
      REQUIRE(*obj == ptr);
      REQUIRE(Py_REFCNT(ptr) == 2);
      REQUIRE(const_cast<const lbann::python::object&>(*obj) == ptr);
      REQUIRE(Py_REFCNT(ptr) == 2);
      REQUIRE(obj->release() == ptr);
      REQUIRE(obj->get() == nullptr);
      REQUIRE(Py_REFCNT(ptr) == 2);
      REQUIRE_NOTHROW(obj.reset());
      REQUIRE(Py_REFCNT(ptr) == 2);
      Py_DECREF(ptr);
      Py_DECREF(ptr);
      REQUIRE_NOTHROW(lbann::python::check_error());
    }

    SECTION("Copy constructor") {
      PyObject* ptr = Py_BuildValue("(i,d,s)", 98, 7.6, "54");
      std::unique_ptr<lbann::python::object> obj1(new lbann::python::object(ptr));
      std::unique_ptr<lbann::python::object> obj2;
      REQUIRE_NOTHROW(obj2.reset(new lbann::python::object(*obj1)));
      REQUIRE(*obj1 == ptr);
      REQUIRE(*obj2 == ptr);
      REQUIRE(Py_REFCNT(ptr) == 2);
      obj1.reset();
      REQUIRE(Py_REFCNT(ptr) == 1);
      obj2.reset();
      REQUIRE_NOTHROW(lbann::python::check_error());
    }

    SECTION("Copy assignment operator") {
      PyObject* ptr1 = Py_BuildValue("(i,d,s)", 1, 2., "3");
      PyObject* ptr2 = Py_BuildValue("(i,d,s)", 4, 5., "6");
      Py_INCREF(ptr1);
      Py_INCREF(ptr2);
      REQUIRE((Py_REFCNT(ptr1) == 2 && Py_REFCNT(ptr2) == 2));
      std::unique_ptr<lbann::python::object> obj1(new lbann::python::object(ptr1));
      std::unique_ptr<lbann::python::object> obj2(new lbann::python::object(ptr2));
      REQUIRE_NOTHROW(*obj2 = *obj1);
      REQUIRE(*obj1 == ptr1);
      REQUIRE(*obj2 == ptr1);
      REQUIRE((Py_REFCNT(ptr1) == 3 && Py_REFCNT(ptr2) == 1));
      obj1.reset();
      REQUIRE((Py_REFCNT(ptr1) == 2 && Py_REFCNT(ptr2) == 1));
      obj2.reset();
      Py_DECREF(ptr1);
      Py_DECREF(ptr2);
      REQUIRE_NOTHROW(lbann::python::check_error());
    }

    SECTION("Move constructor") {
      PyObject* ptr = Py_BuildValue("(i,d,s)", 987, 65.4, "three two one");
      std::unique_ptr<lbann::python::object> obj1(new lbann::python::object(ptr));
      std::unique_ptr<lbann::python::object> obj2;
      REQUIRE_NOTHROW(obj2.reset(new lbann::python::object(std::move(*obj1))));
      REQUIRE(*obj1 == nullptr);
      REQUIRE(*obj2 == ptr);
      REQUIRE(Py_REFCNT(ptr) == 1);
      obj1.reset();
      REQUIRE(Py_REFCNT(ptr) == 1);
      obj2.reset();
      REQUIRE_NOTHROW(lbann::python::check_error());
    }

    SECTION("Move assignment operator") {
      PyObject* ptr1 = Py_BuildValue("(i,d,s)", 9, 8., "7");
      PyObject* ptr2 = Py_BuildValue("(i,d,s)", 6, 5., "4");
      Py_INCREF(ptr1);
      Py_INCREF(ptr2);
      REQUIRE((Py_REFCNT(ptr1) == 2 && Py_REFCNT(ptr2) == 2));
      std::unique_ptr<lbann::python::object> obj1(new lbann::python::object(ptr1));
      std::unique_ptr<lbann::python::object> obj2(new lbann::python::object(ptr2));
      REQUIRE_NOTHROW(*obj2 = std::move(*obj1));
      REQUIRE(*obj1 == nullptr);
      REQUIRE(*obj2 == ptr1);
      REQUIRE((Py_REFCNT(ptr1) == 2 && Py_REFCNT(ptr2) == 1));
      obj1.reset();
      REQUIRE((Py_REFCNT(ptr1) == 2 && Py_REFCNT(ptr2) == 1));
      obj2.reset();
      Py_DECREF(ptr1);
      Py_DECREF(ptr2);
      REQUIRE_NOTHROW(lbann::python::check_error());
    }

    SECTION("str") {
      SECTION("Empty string"){
        std::unique_ptr<lbann::python::object> obj;
        REQUIRE_NOTHROW(obj.reset(new lbann::python::object("")));
        REQUIRE(static_cast<std::string>(*obj).empty());
      }
      SECTION("Non-empty string"){
        std::unique_ptr<lbann::python::object> obj;
        REQUIRE_NOTHROW(obj.reset(new lbann::python::object("one two three")));
        REQUIRE(static_cast<std::string>(*obj) == "one two three");
      }
    }

    SECTION("int") {
      SECTION("Zero") {
        std::unique_ptr<lbann::python::object> obj;
        REQUIRE_NOTHROW(obj.reset(new lbann::python::object(0l)));
        REQUIRE(static_cast<long>(*obj) == 0l);
      }
      SECTION("Positive") {
        std::unique_ptr<lbann::python::object> obj;
        REQUIRE_NOTHROW(obj.reset(new lbann::python::object(123l)));
        REQUIRE(static_cast<long>(*obj) == 123l);
      }
      SECTION("Negative") {
        std::unique_ptr<lbann::python::object> obj;
        REQUIRE_NOTHROW(obj.reset(new lbann::python::object(-321l)));
        REQUIRE(static_cast<long>(*obj) == -321l);
      }
    }

    SECTION("float") {
      SECTION("Zero") {
        std::unique_ptr<lbann::python::object> obj;
        REQUIRE_NOTHROW(obj.reset(new lbann::python::object(0.0)));
        REQUIRE(static_cast<double>(*obj) == 0.0);
      }
      SECTION("Positive") {
        std::unique_ptr<lbann::python::object> obj;
        REQUIRE_NOTHROW(obj.reset(new lbann::python::object(3.21)));
        REQUIRE(static_cast<double>(*obj) == 3.21);
      }
      SECTION("Negative") {
        std::unique_ptr<lbann::python::object> obj;
        REQUIRE_NOTHROW(obj.reset(new lbann::python::object(-12.3)));
        REQUIRE(static_cast<double>(*obj) == -12.3);
      }
    }

  }

}
#endif // LBANN_HAS_PYTHON
