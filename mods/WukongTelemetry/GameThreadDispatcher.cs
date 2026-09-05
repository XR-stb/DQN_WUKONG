using System;
using System.Reflection;
using UnrealEngine.Runtime;

namespace WukongTelemetry
{
    /// <summary>
    /// AOT-safe bridge to the loader's internal, persistent game-thread queue.
    /// Reflection and delegate binding happen once during Mod initialization.
    /// </summary>
    internal sealed class GameThreadDispatcher
    {
        private const string HelperTypeName = "UnrealEngine.GameThreadHelper";

        private readonly MethodInfo _runMethod;
        private readonly object[] _arguments;

        public GameThreadDispatcher(Action callback)
        {
            var runtimeAssembly = typeof(UObject).Assembly;
            var helperType = runtimeAssembly.GetType(HelperTypeName, throwOnError: true);
            var callbackType = helperType.GetNestedType(
                "FSimpleDelegate", BindingFlags.Public | BindingFlags.NonPublic);
            if (callbackType == null)
                throw new MissingMemberException(HelperTypeName, "FSimpleDelegate");

            _runMethod = helperType.GetMethod(
                "Run",
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                types: new[] { callbackType },
                modifiers: null)
                ?? throw new MissingMethodException(HelperTypeName, "Run");

            var boundCallback = Delegate.CreateDelegate(
                callbackType,
                callback.Target,
                callback.Method,
                throwOnBindFailure: true);
            _arguments = new object[] { boundCallback };
        }

        public void Invoke()
        {
            _runMethod.Invoke(null, _arguments);
        }
    }
}
