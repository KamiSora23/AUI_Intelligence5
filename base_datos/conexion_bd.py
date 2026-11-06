import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Cargar la configuración propia del proyecto por encima de variables globales
# de Windows que pueden pertenecer a otras aplicaciones.
load_dotenv(override=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "bd_deteccion_fraccionamiento")
DB_MODE = os.getenv("DB_MODE", "mariadb").strip().lower()

# El dialecto mysql+mysqlconnector es compatible con MariaDB.
DATABASE_URL_PRINCIPAL = (
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

_engine = None
def _crear_engine_internal():
    global _engine
    if _engine is not None:
        return _engine

    if DB_MODE not in {"mariadb", "mysql"}:
        raise RuntimeError(
            "DB_MODE debe ser 'mariadb' o 'mysql'. "
            "La aplicación trabaja exclusivamente con MySQL/MariaDB."
        )

    try:
        engine = create_engine(
            DATABASE_URL_PRINCIPAL,
            pool_pre_ping=True,
            echo=False,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        with engine.begin() as conn:
            if "flujo_revision" not in {
                columna["name"]
                for columna in inspect(engine).get_columns("configuracion_sistema")
            }:
                conn.execute(
                    text(
                        "ALTER TABLE configuracion_sistema "
                        "ADD COLUMN flujo_revision VARCHAR(20) "
                        "DEFAULT 'Automático'"
                    )
                )
            if "etiqueta_referencia" not in {
                columna["name"]
                for columna in inspect(engine).get_columns("resultados_anomalias")
            }:
                conn.execute(
                    text(
                        "ALTER TABLE resultados_anomalias "
                        "ADD COLUMN etiqueta_referencia TINYINT(1) NULL"
                    )
                )
            # Atipicidad del Isolation Forest: ordena las compras que empatan
            # en puntaje de riesgo.
            if "atipicidad_ia" not in {
                columna["name"]
                for columna in inspect(engine).get_columns("resultados_anomalias")
            }:
                conn.execute(
                    text(
                        "ALTER TABLE resultados_anomalias "
                        "ADD COLUMN atipicidad_ia DOUBLE NULL"
                    )
                )
            # Ventana con la que se procesó cada lote; la configuración puede
            # cambiar después y no describiría los resultados ya guardados.
            if "ventana_dias" not in {
                columna["name"]
                for columna in inspect(engine).get_columns("historial_cargas")
            }:
                conn.execute(
                    text(
                        "ALTER TABLE historial_cargas "
                        "ADD COLUMN ventana_dias INT NULL"
                    )
                )
            try:
                conv_cols = inspect(engine).get_columns("convocatorias_procesadas")
                for col in conv_cols:
                    if col["name"] == "entidad_ruc":
                        conn.execute(
                            text(
                                "ALTER TABLE convocatorias_procesadas "
                                "MODIFY COLUMN entidad_ruc VARCHAR(20) NOT NULL"
                            )
                        )
                        break
            except Exception:
                pass
        _engine = engine
        print("[BD] Conexión exitosa a MariaDB.")
    except Exception as e:
        raise ConnectionError(
            f"No se pudo conectar a MySQL/MariaDB en "
            f"{DB_HOST}:{DB_PORT}/{DB_NAME}."
        ) from e

    return _engine

if HAS_STREAMLIT:
    @st.cache_resource
    def obtener_engine():
        return _crear_engine_internal()
else:
    def obtener_engine():
        return _crear_engine_internal()

def _inicializar_sqlite_fallback(engine):
    """Inicializa la estructura relacional y los datos por defecto en SQLite en caso de fallback."""
    ddl_sqlite = """
    CREATE TABLE IF NOT EXISTS usuarios (
        id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre_usuario TEXT NOT NULL UNIQUE,
        clave_encriptada TEXT NOT NULL,
        nombre_completo TEXT NOT NULL,
        correo TEXT NULL,
        rol TEXT DEFAULT 'Auditor',
        estado INTEGER DEFAULT 1,
        fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS logs_sistema (
        id_log INTEGER PRIMARY KEY AUTOINCREMENT,
        id_usuario INTEGER NULL,
        accion TEXT NOT NULL,
        modulo TEXT NOT NULL,
        ip_origen TEXT DEFAULT '127.0.0.1',
        detalle TEXT,
        fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario)
    );

    CREATE TABLE IF NOT EXISTS configuracion_sistema (
        id_config INTEGER PRIMARY KEY AUTOINCREMENT,
        frecuencia_contaminacion REAL DEFAULT 0.030,
        umbral_similitud_nlp REAL DEFAULT 0.50,
        ventana_dias INTEGER DEFAULT 30,
        monto_uit_vigente REAL DEFAULT 5150.00,
        ruta_almacenamiento_pdf TEXT DEFAULT './reportes_pdf/',
        notificar_riesgo_alto INTEGER DEFAULT 1,
        umbral_alerta_notificacion REAL DEFAULT 0.85,
        flujo_revision TEXT DEFAULT 'Automático',
        fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS modelos_ia (
        id_modelo INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre_modelo TEXT NOT NULL,
        fecha_entrenamiento DATETIME DEFAULT CURRENT_TIMESTAMP,
        metrica_precision REAL NULL,
        metrica_recall REAL NULL,
        metrica_f1 REAL NULL,
        metrica_roc_auc REAL NULL,
        estado_activo INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS historial_cargas (
        id_carga INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre_archivo TEXT NOT NULL,
        total_registros INTEGER NOT NULL,
        fecha_carga DATETIME DEFAULT CURRENT_TIMESTAMP,
        id_usuario INTEGER NULL,
        FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario)
    );

    CREATE TABLE IF NOT EXISTS convocatorias_procesadas (
        id_convocatoria INTEGER PRIMARY KEY AUTOINCREMENT,
        id_carga INTEGER NOT NULL,
        entidad_ruc TEXT NOT NULL,
        entidad_nombre TEXT NOT NULL,
        descripcion_item TEXT NOT NULL,
        monto_referencial REAL NOT NULL,
        fecha_convocatoria DATE NOT NULL,
        codigo_cubso TEXT NULL,
        tipo_proceso TEXT DEFAULT 'Igual o Menor a 8 UIT',
        FOREIGN KEY (id_carga) REFERENCES historial_cargas(id_carga)
    );

    CREATE TABLE IF NOT EXISTS resultados_anomalias (
        id_resultado INTEGER PRIMARY KEY AUTOINCREMENT,
        id_convocatoria INTEGER NOT NULL,
        id_modelo INTEGER NULL,
        similitud_nlp REAL NOT NULL,
        monto_acumulado_30d REAL NOT NULL,
        frecuencia_compras_30d INTEGER NOT NULL,
        dias_desde_compra_anterior INTEGER NOT NULL,
        score_reglas_negocio REAL DEFAULT 0,
        bono_ia REAL DEFAULT 0,
        puntaje_riesgo REAL NOT NULL,
        nivel_riesgo TEXT DEFAULT 'Bajo',
        es_anomalia INTEGER NOT NULL,
        explicacion_reglas TEXT NULL,
        estado_revision TEXT DEFAULT 'Pendiente',
        comentario_auditor TEXT NULL,
        id_usuario_auditor INTEGER NULL,
        fecha_revision DATETIME NULL,
        FOREIGN KEY (id_convocatoria) REFERENCES convocatorias_procesadas(id_convocatoria)
    );

    CREATE TABLE IF NOT EXISTS datasets_evaluacion (
        id_dataset INTEGER PRIMARY KEY AUTOINCREMENT,
        tipo_dataset TEXT NOT NULL,
        cantidad_registros INTEGER NOT NULL,
        fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    with engine.begin() as conn:
        for stmt in ddl_sqlite.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))

        # Mantener compatibles las bases SQLite creadas por versiones anteriores.
        # CREATE TABLE IF NOT EXISTS no agrega columnas nuevas a tablas existentes.
        migraciones_columnas = {
            "usuarios": {
                "correo": "TEXT NULL",
            },
            "convocatorias_procesadas": {
                "tipo_proceso": "TEXT DEFAULT 'Igual o Menor a 8 UIT'",
            },
            "configuracion_sistema": {
                "flujo_revision": "TEXT DEFAULT 'Automático'",
            },
            "resultados_anomalias": {
                "score_reglas_negocio": "REAL DEFAULT 0",
                "bono_ia": "REAL DEFAULT 0",
                "atipicidad_ia": "REAL NULL",
                "nivel_riesgo": "TEXT DEFAULT 'Bajo'",
                "explicacion_reglas": "TEXT NULL",
            },
        }

        for tabla, columnas in migraciones_columnas.items():
            columnas_existentes = {
                fila[1]
                for fila in conn.execute(text(f'PRAGMA table_info("{tabla}")')).fetchall()
            }
            for columna, definicion in columnas.items():
                if columna not in columnas_existentes:
                    conn.execute(
                        text(f'ALTER TABLE "{tabla}" ADD COLUMN "{columna}" {definicion}')
                    )

        # Insertar datos por defecto si la tabla de usuarios está vacía
        res = conn.execute(text("SELECT COUNT(*) FROM usuarios")).scalar()
        if res == 0:
            import bcrypt
            # Hash para 'admin123'
            clave_admin = bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode('utf-8')
            # Hash para 'auditor123'
            clave_auditor = bcrypt.hashpw(b"auditor123", bcrypt.gensalt()).decode('utf-8')

            conn.execute(text("""
                INSERT INTO usuarios (nombre_usuario, clave_encriptada, nombre_completo, rol, estado)
                VALUES ('admin', :clv1, 'Administrador Principal SEACE', 'Administrador', 1)
            """), {"clv1": clave_admin})

            conn.execute(text("""
                INSERT INTO usuarios (nombre_usuario, clave_encriptada, nombre_completo, rol, estado)
                VALUES ('auditor', :clv2, 'Auditor de Contrataciones OSCE', 'Auditor', 1)
            """), {"clv2": clave_auditor})

        res_cfg = conn.execute(text("SELECT COUNT(*) FROM configuracion_sistema")).scalar()
        if res_cfg == 0:
            conn.execute(text("""
                INSERT INTO configuracion_sistema (frecuencia_contaminacion, umbral_similitud_nlp, ventana_dias, monto_uit_vigente, ruta_almacenamiento_pdf, notificar_riesgo_alto, umbral_alerta_notificacion)
                VALUES (0.030, 0.50, 30, 5150.00, './reportes_pdf/', 1, 0.85)
            """))

        res_mod = conn.execute(text("SELECT COUNT(*) FROM modelos_ia")).scalar()
        if res_mod == 0:
            conn.execute(text("""
                INSERT INTO modelos_ia (nombre_modelo, metrica_precision, metrica_recall, metrica_f1, metrica_roc_auc, estado_activo)
                VALUES ('Isolation Forest v1.0 (Baseline)', 0.8850, 0.8420, 0.8629, 0.9110, 1)
            """))

def es_modo_sqlite():
    """Compatibilidad con la interfaz: el modo SQLite está deshabilitado."""
    return False

def obtener_session():
    """Retorna una sesión SQLAlchemy limpia."""
    engine = obtener_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def obtener_usuario_activo_por_id(id_usuario):
    """Recupera una cuenta activa para validar la restauración de una sesión."""
    session = obtener_session()
    try:
        resultado = session.execute(
            text(
                """
                SELECT id_usuario, nombre_usuario, nombre_completo, correo, rol, estado
                FROM usuarios
                WHERE id_usuario = :id_usuario AND estado = 1
                """
            ),
            {"id_usuario": id_usuario},
        ).fetchone()
        return dict(resultado._mapping) if resultado else None
    except Exception as error:
        print(f"Error al recuperar usuario activo: {error}")
        return None
    finally:
        session.close()


def registrar_sesion_web(id_sesion, id_usuario, fecha_expiracion):
    """Registra una sesión firmada para permitir su revocación inmediata."""
    _asegurar_tabla_sesiones_web()
    session = obtener_session()
    try:
        session.execute(
            text(
                """
                INSERT INTO sesiones_web
                    (id_sesion, id_usuario, fecha_expiracion, revocada)
                VALUES
                    (:id_sesion, :id_usuario, :fecha_expiracion, 0)
                """
            ),
            {
                "id_sesion": id_sesion,
                "id_usuario": id_usuario,
                "fecha_expiracion": fecha_expiracion,
            },
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def sesion_web_activa(id_sesion, id_usuario):
    """Comprueba que una sesión exista, no esté revocada y no haya vencido."""
    _asegurar_tabla_sesiones_web()
    session = obtener_session()
    try:
        cantidad = session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM sesiones_web
                WHERE id_sesion = :id_sesion
                  AND id_usuario = :id_usuario
                  AND revocada = 0
                  AND fecha_expiracion > CURRENT_TIMESTAMP
                """
            ),
            {"id_sesion": id_sesion, "id_usuario": id_usuario},
        ).scalar()
        return bool(cantidad)
    except Exception as error:
        print(f"Error al validar sesión web: {error}")
        return False
    finally:
        session.close()


def revocar_sesion_web(id_sesion):
    """Invalida una sesión aunque su cookie todavía exista en el navegador."""
    _asegurar_tabla_sesiones_web()
    session = obtener_session()
    try:
        session.execute(
            text(
                """
                UPDATE sesiones_web
                SET revocada = 1
                WHERE id_sesion = :id_sesion
                """
            ),
            {"id_sesion": id_sesion},
        )
        session.commit()
    except Exception as error:
        session.rollback()
        print(f"Error al revocar sesión web: {error}")
    finally:
        session.close()


def _asegurar_tabla_sesiones_web():
    """Crea la tabla de sesiones en MariaDB o en el respaldo SQLite."""
    with obtener_engine().begin() as conexion:
        conexion.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS sesiones_web (
                    id_sesion VARCHAR(64) PRIMARY KEY,
                    id_usuario INTEGER NOT NULL,
                    fecha_expiracion DATETIME NOT NULL,
                    revocada INTEGER NOT NULL DEFAULT 0,
                    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )


def autenticar_usuario(nombre_usuario, clave):
    """
    Autentica un usuario contra la base de datos comparando la contraseña cifrada con Bcrypt.
    Retorna un diccionario con los datos del usuario si la autenticación es exitosa, o None en caso contrario.
    """
    import bcrypt
    session = obtener_session()
    try:
        res = session.execute(
            text("SELECT * FROM usuarios WHERE nombre_usuario = :usr AND estado = 1"),
            {"usr": nombre_usuario}
        ).fetchone()

        if not res:
            return None

        usr_dict = dict(res._mapping)
        hash_guardado = str(usr_dict.get('clave_encriptada', ''))

        # Comprobación Bcrypt
        try:
            if hash_guardado and bcrypt.checkpw(clave.encode('utf-8'), hash_guardado.encode('utf-8')):
                return usr_dict
        except Exception:
            pass

        return None
    except Exception as e:
        print(f"Error en autenticar_usuario: {e}")
        return None
    finally:
        session.close()
